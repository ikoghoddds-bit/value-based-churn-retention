"""Reproducible churn benchmark. Run from the repository root."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
SEED = 42
# Demographics omitted from prediction; retained separately for subgroup audit.
EXCLUDE = ['customerID','Churn','gender','SeniorCitizen','Partner','Dependents']

def load_data():
    d = pd.read_csv(ROOT/'data/telco.csv')
    d['TotalCharges'] = pd.to_numeric(d.TotalCharges, errors='coerce')
    assert d.customerID.is_unique and d.Churn.isin(['Yes','No']).all()
    assert d.MonthlyCharges.ge(0).all()
    return d

from src.business import campaign, prioritize

def metrics(y,p,threshold=.5):
    z=p>=threshold
    return dict(roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),
        brier=brier_score_loss(y,p),precision=precision_score(y,z,zero_division=0),
        recall=recall_score(y,z,zero_division=0),f1=f1_score(y,z,zero_division=0),f2=fbeta_score(y,z,beta=2,zero_division=0))

def main():
    OUT.mkdir(exist_ok=True)
    d=load_data(); X=d.drop(columns=EXCLUDE); y=d.Churn.eq('Yes').astype(int)
    train, rest=train_test_split(d.index,test_size=.4,stratify=y,random_state=SEED)
    val,test=train_test_split(rest,test_size=.5,stratify=y.loc[rest],random_state=SEED)
    nums=X.select_dtypes('number').columns.tolist(); cats=X.select_dtypes('object').columns.tolist()
    def prep():
        return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    candidates={'baseline':DummyClassifier(strategy='prior'),
        'logistic':LogisticRegression(C=1,max_iter=2000,random_state=SEED)}
    fitted={}; rows=[]
    for name,est in candidates.items():
        pipe=Pipeline([('prep',prep()),('model',est)])
        # Calibration folds and their preprocessing see only training data.
        model=pipe if name=='baseline' else CalibratedClassifierCV(pipe,method='sigmoid',cv=5)
        model.fit(X.loc[train],y.loc[train]); fitted[name]=model
        rows.append({'model':name,**metrics(y.loc[val],model.predict_proba(X.loc[val])[:,1])})
    comparison=pd.DataFrame(rows).set_index('model'); comparison.to_csv(OUT/'validation_metrics.csv')
    # Keep the requested simple logistic model. Baseline is a sanity check,
    # not a model competition. Select a recall-first threshold on validation.
    winner='logistic'; model=fitted[winner]
    pv=model.predict_proba(X.loc[val])[:,1]
    thresholds=np.round(np.arange(0.,1.001,.01),2)
    tradeoffs=[]
    for t in thresholds:
        pred=pv>=t
        tn,fp,fn,tp=confusion_matrix(y.loc[val],pred,labels=[0,1]).ravel()
        # Illustrative asymmetry: missing a churner costs 5x an unnecessary call.
        tradeoffs.append({'threshold':t,**metrics(y.loc[val],pv,t),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'diagnostic_cost_5_to_1':int(fp)*20+int(fn)*100})
    threshold_table=pd.DataFrame(tradeoffs)
    threshold=float(threshold_table.loc[threshold_table.recall>=.85,'threshold'].max())
    threshold_table.to_csv(OUT/'validation_threshold_tradeoffs.csv',index=False)
    # All-positive screening trivially maximizes recall. Taking the highest
    # threshold meeting 85% validation recall limits unnecessary calls.
    p=model.predict_proba(X.loc[test])[:,1]; yt=y.loc[test]
    result=metrics(yt,p,threshold)
    scores=d.loc[test,['customerID','MonthlyCharges','Contract','tenure','gender','SeniorCitizen']].copy()
    scores['probability']=p; scores['actual_churn']=yt
    action=prioritize(scores.drop(columns=['actual_churn','gender','SeniorCitizen']),threshold)
    action.to_csv(OUT/'prioritized_call_list.csv',index=False)
    action[action.churn_flag].to_csv(OUT/'flagged_customers.csv',index=False)
    scores.to_csv(OUT/'test_scores.csv',index=False)
    pd.DataFrame({'row':d.index,'customerID':d.customerID,'split':['train' if i in train else 'validation' if i in val else 'test' for i in d.index]}).to_csv(OUT/'splits.csv',index=False)
    joblib.dump({'model':model,'features':X.columns.tolist(),'threshold':threshold},OUT/'model.joblib')
    policies=[]
    for strategy in ['value','risk','random']:
        selected=campaign(scores,strategy=strategy)
        # Counterfactual accounting using observed labels, still assumes 25% save rate.
        oracle=(selected.actual_churn*selected.MonthlyCharges*6*.25*.7-20).sum()
        policies.append(dict(strategy=strategy,contacts=len(selected),expected_saved_revenue=selected.expected_saved_revenue.sum(),expected_net_value=selected.expected_net_value.sum(),label_based_scenario_net=oracle))
    pd.DataFrame(policies).to_csv(OUT/'policy_comparison.csv',index=False)
    queue=campaign(scores); queue.drop(columns=['actual_churn','gender','SeniorCitizen']).to_csv(OUT/'retention_queue.csv',index=False)
    sens=[]
    for s in [.1,.2,.25,.3,.4]:
        for c in [10,20,40,60]:
            q=campaign(scores,success=s,cost=c)
            sens.append(dict(success=s,cost=c,contacts=len(q),expected_net=q.expected_net_value.sum()))
    pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False)
    # Paired row bootstrap AUC interval: fixed model; not training uncertainty.
    rng=np.random.default_rng(SEED); auc=[]
    for _ in range(1000):
        ix=rng.integers(0,len(yt),len(yt)); yy=yt.to_numpy()[ix]
        if len(np.unique(yy))==2: auc.append(roc_auc_score(yy,p[ix]))
    result['auc_ci95']=np.quantile(auc,[.025,.975]).tolist()
    result.update(winner=winner,threshold=threshold,rows=len(d),test_rows=len(test),
        churn_rate=float(y.mean()),test_churn_rate=float(yt.mean()),missing_total_charges=int(d.TotalCharges.isna().sum()),
        monthly_revenue_at_risk=float((scores.probability*scores.MonthlyCharges).sum()),
        six_month_revenue_at_risk=float((scores.probability*scores.MonthlyCharges*6).sum()),
        campaign=policies[0],confusion_matrix=confusion_matrix(yt,p>=threshold).tolist(),
        flagged_customers=int((p>=threshold).sum()),validation_recall=float(recall_score(y.loc[val],pv>=threshold)),
        test_at_threshold_0_5=metrics(yt,p,.5),
        diagnostic_cost_5_to_1=float(((yt==0)&(p>=threshold)).sum()*20+((yt==1)&(p<threshold)).sum()*100),
        data_sha256=hashlib.sha256((ROOT/'data/telco.csv').read_bytes()).hexdigest())
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    audit=[]
    for col in ['gender','SeniorCitizen']:
        for group,g in scores.groupby(col):
            audit.append(dict(attribute=col,group=str(group),n=len(g),observed_rate=g.actual_churn.mean(),predicted_rate=g.probability.mean(),brier=brier_score_loss(g.actual_churn,g.probability),recall=recall_score(g.actual_churn,g.probability>=threshold,zero_division=0)))
    pd.DataFrame(audit).to_csv(OUT/'subgroup_audit.csv',index=False)
    imp=permutation_importance(model,X.loc[test],yt,n_repeats=5,random_state=SEED,scoring='roc_auc',n_jobs=1)
    importance=pd.DataFrame({'feature':X.columns,'auc_drop':imp.importances_mean,'std':imp.importances_std}).sort_values('auc_drop',ascending=False)
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    RocCurveDisplay.from_predictions(yt,p,ax=ax[0,0],name=winner,curve_kwargs={'color':'#087e8b'}); ax[0,0].set_title('Held-out ROC curve')
    PrecisionRecallDisplay.from_predictions(yt,p,ax=ax[0,1],name=winner,color='#087e8b'); ax[0,1].axhline(yt.mean(),ls='--',color='gray'); ax[0,1].set_title('Precision and recall')
    frac,mean=calibration_curve(yt,p,n_bins=10,strategy='quantile'); ax[0,2].plot(mean,frac,'o-',color='#087e8b'); ax[0,2].plot([0,1],[0,1],'--',color='gray'); ax[0,2].set(xlabel='Mean predicted risk',ylabel='Observed churn fraction',title='Probability calibration')
    importance.head(8).iloc[::-1].plot.barh(x='feature',y='auc_drop',legend=False,ax=ax[1,0],color='#087e8b'); ax[1,0].set_title('Permutation importance (association)')
    ranked=scores.sort_values('probability',ascending=False); gain=ranked.actual_churn.cumsum()/yt.sum(); ax[1,1].plot(np.arange(1,len(gain)+1)/len(gain),gain,color='#087e8b'); ax[1,1].plot([0,1],[0,1],'--',color='gray'); ax[1,1].set(xlabel='Fraction contacted',ylabel='Fraction of churners captured',title='Cumulative gains')
    pd.DataFrame(policies).plot.bar(x='strategy',y='label_based_scenario_net',ax=ax[1,2],legend=False,color='#e2ab48',rot=0); ax[1,2].set(ylabel='USD, assumed campaign economics',title='Label-based scenario (not actual profit)')
    fig.savefig(OUT/'evaluation.png',dpi=150); plt.close(fig)
    # Compare business screening costs at several assumed FN:FP cost ratios.
    costs=[]
    for ratio in [1,3,5,10]:
        objective=threshold_table.fp*20+threshold_table.fn*20*ratio
        cost_t=float(threshold_table.loc[objective.idxmin(),'threshold'])
        for name,t in [('recall_first',threshold),('default_0_5',.5),('validation_cost_min',cost_t)]:
            tn,fp,fn,tp=confusion_matrix(yt,p>=t,labels=[0,1]).ravel()
            costs.append(dict(fn_to_fp_ratio=ratio,policy=name,threshold=t,fp=int(fp),fn=int(fn),diagnostic_cost=int(fp)*20+int(fn)*20*ratio))
    pd.DataFrame(costs).to_csv(OUT/'error_cost_sensitivity.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
