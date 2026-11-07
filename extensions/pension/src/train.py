"""python -m src.train: embargoed temporal evaluation of synthetic pension data."""
from pathlib import Path
import json,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score,fbeta_score,confusion_matrix,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
from src.business import FEATURES,NUMERIC,validate,attach_financials,campaign
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';SEED=42

def split_data(d):
    dates=pd.to_datetime(d.snapshot_date);split=pd.Series('embargo',index=d.index)
    split.loc[dates<='2024-06-30']='train'
    split.loc[dates.between('2024-10-01','2025-03-31')]='validation'
    split.loc[dates>='2025-07-01']='test'
    return split

def metrics(y,p,t=.5):
    return dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),brier=brier_score_loss(y,p),precision=precision_score(y,p>=t,zero_division=0),recall=recall_score(y,p>=t,zero_division=0),f2=fbeta_score(y,p>=t,beta=2,zero_division=0))

def main():
    OUT.mkdir(exist_ok=True);raw=pd.read_csv(ROOT/'data/synthetic_members.csv');d=validate(raw)
    y=raw.churn_90d;X=d[FEATURES];split=split_data(d)
    tr=d.index[split=='train'];va=d.index[split=='validation'];te=d.index[split=='test']
    assert pd.to_datetime(raw.loc[tr,'outcome_window_end']).max()<pd.to_datetime(d.loc[va,'snapshot_date']).min()
    assert pd.to_datetime(raw.loc[va,'outcome_window_end']).max()<pd.to_datetime(d.loc[te,'snapshot_date']).min()
    def prep():return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['employment_status'])])
    candidates={'prior':DummyClassifier(strategy='prior'),'logistic':LogisticRegression(max_iter=2000,C=1,random_state=SEED),'boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=10,l2_regularization=10,random_state=SEED)}
    models={};comparison=[]
    fit=tr[pd.to_datetime(d.loc[tr,'snapshot_date'])<='2023-12-31'];cal=tr[pd.to_datetime(d.loc[tr,'snapshot_date'])>'2024-03-31']
    assert pd.to_datetime(raw.loc[fit,'outcome_window_end']).max()<pd.to_datetime(d.loc[cal,'snapshot_date']).min()
    for name,est in candidates.items():
        base=Pipeline([('prep',prep()),('model',est)]).fit(X.loc[fit],y.loc[fit])
        model=base if name=='prior' else CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid').fit(X.loc[cal],y.loc[cal])
        models[name]=model;comparison.append({'model':name,**metrics(y.loc[va],model.predict_proba(X.loc[va])[:,1])})
    comparison=pd.DataFrame(comparison).set_index('model');comparison.to_csv(OUT/'validation_metrics.csv')
    winner=comparison.drop(index='prior').brier.idxmin();model=models[winner];pv=model.predict_proba(X.loc[va])[:,1]
    ts=np.round(np.arange(.05,.81,.01),2);threshold=float(max(ts,key=lambda t:fbeta_score(y.loc[va],pv>=t,beta=2,zero_division=0)))
    p=model.predict_proba(X.loc[te])[:,1];yt=y.loc[te];s=attach_financials(d.loc[te],p,threshold)
    s['actual_churn_90d']=yt;s['actual_outflow_90d']=raw.loc[te,'actual_outflow_90d'];s.to_csv(OUT/'member_predictions.csv',index=False)
    flags=s[s.churn_flag].sort_values('churn_probability_90d',ascending=False);flags.to_csv(OUT/'flagged_members.csv',index=False)
    q=campaign(s);q.drop(columns=['actual_churn_90d','actual_outflow_90d']).to_csv(OUT/'retention_priority.csv',index=False)
    joblib.dump({'model':model,'features':FEATURES,'threshold':threshold,'version':'2.0-pension-synthetic'},OUT/'model.joblib')
    stage=split.copy();stage.loc[tr]='internal_embargo';stage.loc[fit]='base_fit';stage.loc[cal]='calibration'
    raw[['member_id','snapshot_date','outcome_window_end']].assign(split=split,model_stage=stage).to_csv(OUT/'splits.csv',index=False)
    policy=[]
    for strategy in ['value','risk','random']:
        c=campaign(s,strategy=strategy)
        policy.append(dict(policy=strategy,contacts=len(c),expected_assets_retained=c.expected_assets_retained.sum(),expected_fee_revenue_preserved=c.expected_fee_revenue_preserved.sum(),expected_net_contribution=c.expected_net_contribution.sum(),label_based_scenario_net=(c.actual_churn_90d*c.annual_fee_revenue_if_member_leaves*.25*.7-25).sum()))
    pd.DataFrame(policy).to_csv(OUT/'policy_comparison.csv',index=False)
    sensitivity=[]
    for save in [.1,.2,.25,.3,.4]:
        for cost in [10,25,50,100]:
            c=campaign(s,save_rate=save,cost=cost);sensitivity.append(dict(save_rate=save,cost=cost,contacts=len(c),expected_net_contribution=c.expected_net_contribution.sum()))
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    rng=np.random.default_rng(SEED);aucs=[]
    for _ in range(1000):
        ix=rng.integers(0,len(yt),len(yt));yy=yt.to_numpy()[ix]
        if len(np.unique(yy))==2:aucs.append(roc_auc_score(yy,p[ix]))
    result={**metrics(yt,p,threshold),'auc_ci95':np.quantile(aucs,[.025,.975]).tolist(),'winner':winner,'threshold':threshold,'rows':len(raw),'train_rows':len(tr),'base_fit_rows':len(fit),'calibration_rows':len(cal),'internal_embargo_rows':len(tr)-len(fit)-len(cal),'validation_rows':len(va),'test_rows':len(te),'embargo_rows':int((split=='embargo').sum()),'test_churn_rate':float(yt.mean()),'flagged_members':len(flags),'test_total_balance':float(s.account_balance.sum()),'flagged_balance_if_all_leave':float(flags.amount_if_member_leaves.sum()),'expected_outflow_90d':float(s.expected_outflow_90d.sum()),'expected_annual_fee_revenue_at_risk':float(s.expected_annual_fee_revenue_at_risk.sum()),'observed_synthetic_outflow_90d':float(s.actual_outflow_90d.sum()),'campaign':policy[0],'confusion_matrix':confusion_matrix(yt,p>=threshold).tolist(),'data_sha256':hashlib.sha256((ROOT/'data/synthetic_members.csv').read_bytes()).hexdigest()}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    audit=[];groups=pd.cut(d.loc[te,'age'],[0,35,45,60],labels=['25–35','36–45','46–60'])
    for group in groups.dropna().unique():
        mask=(groups==group).to_numpy();yy=yt.to_numpy()[mask];pp=p[mask];audit.append(dict(age_group=str(group),n=int(mask.sum()),observed_churn=yy.mean(),predicted_churn=pp.mean(),brier=brier_score_loss(yy,pp),recall=recall_score(yy,pp>=threshold,zero_division=0)))
    pd.DataFrame(audit).to_csv(OUT/'subgroup_audit.csv',index=False)
    imp=permutation_importance(model,X.loc[te],yt,n_repeats=5,random_state=SEED,scoring='roc_auc');importance=pd.DataFrame({'feature':FEATURES,'auc_drop':imp.importances_mean,'std':imp.importances_std}).sort_values('auc_drop',ascending=False);importance.to_csv(OUT/'feature_importance.csv',index=False)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10});fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    RocCurveDisplay.from_predictions(yt,p,ax=ax[0,0],name=winner,curve_kwargs={'color':'#07828b'});ax[0,0].set_title('Held-out temporal ROC')
    PrecisionRecallDisplay.from_predictions(yt,p,ax=ax[0,1],name=winner);ax[0,1].axhline(yt.mean(),color='gray',ls='--');ax[0,1].set_title('Precision-recall versus prevalence')
    frac,mean=calibration_curve(yt,p,n_bins=10,strategy='quantile');ax[0,2].plot(mean,frac,'o-',color='#07828b');ax[0,2].plot([0,1],[0,1],'--',color='gray');ax[0,2].set(xlabel='Predicted risk',ylabel='Observed churn',title='Probability calibration')
    importance.head(7).iloc[::-1].plot.barh(x='feature',y='auc_drop',legend=False,ax=ax[1,0],color='#07828b');ax[1,0].set_title('Permutation importance: associations')
    ranked=s.sort_values('churn_probability_90d',ascending=False);xx=np.arange(1,len(s)+1)/len(s);ax[1,1].plot(xx,ranked.actual_outflow_90d.cumsum()/s.actual_outflow_90d.sum(),color='#07828b');ax[1,1].plot([0,1],[0,1],'--',color='gray');ax[1,1].set(xlabel='Fraction of members ranked by risk',ylabel='Fraction of synthetic outflow captured',title='Outflow capture by risk ranking')
    pd.DataFrame(policy).plot.bar(x='policy',y='label_based_scenario_net',ax=ax[1,2],legend=False,color='#e1ac48',rot=0);ax[1,2].set(ylabel='USD, assumed campaign economics',title='Label-based scenario, not actual impact')
    fig.suptitle('SYNTHETIC PENSION DEMONSTRATION · No real fund performance',fontsize=15);fig.savefig(OUT/'evaluation.png',dpi=150);plt.close(fig)
    pkgs=['numpy','pandas','scikit-learn','scipy','matplotlib','joblib','streamlit','nbformat','nbclient','ipykernel','pytest'];(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':{x:importlib.metadata.version(x) for x in pkgs}},indent=2))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
