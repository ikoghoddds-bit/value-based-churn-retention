"""Local dashboard: streamlit run app.py"""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.business import attach_financials,campaign
from src.score import score_frame
ROOT=Path(__file__).parent
st.set_page_config(page_title='Pension retention | Joseph Ikogho',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp{background:#f5f8fa}h1,h2,h3{color:#123b46}div[data-testid="stMetric"]{background:white;padding:18px;border-radius:12px;border-top:3px solid #07828b}</style>''',unsafe_allow_html=True)
st.caption('JOSEPH IKOGHO · DATA SCIENCE PORTFOLIO / 01')
st.title('Pension member churn and financial exposure')
st.warning('SYNTHETIC DATA DEMONSTRATION. No real members, fund results or validated retention effects.')
st.write('Identify members who may leave, show how much each could take, and plan a measurable retention pilot.')
if not (ROOT/'outputs/metrics.json').exists():st.error('Run python -m src.train first.');st.stop()
m=json.loads((ROOT/'outputs/metrics.json').read_text());base=pd.read_csv(ROOT/'outputs/member_predictions.csv')
with st.sidebar:
    st.header('Decision controls')
    threshold=st.slider('Risk flag threshold',0.,1.,float(m['threshold']),.01)
    withdrawal=st.slider('Share of balance transferred on exit',0.,1.,1.,.05)
    st.caption('Full balance transfer is the synthetic label definition. Lower fractions are sensitivity scenarios.')
    st.header('Retention assumptions')
    save=st.slider('Would-be leavers saved',0.,1.,.25,.05)
    years=st.slider('Fee revenue horizon (years)',1,3,1)
    margin=st.slider('Contribution margin',0.,1.,.7,.05)
    cost=st.number_input('All-in cost per contact ($)',0.,1000.,25.,5.)
    capacity=st.slider('Maximum contacts',0,len(base),200)
    st.caption('Queue includes only flagged members with positive expected contribution. Revenue assumes a constant balance and fee rate over the horizon.')
scores=attach_financials(base,base.churn_probability_90d,threshold,withdrawal)
flags=scores[scores.churn_flag].sort_values('churn_probability_90d',ascending=False)
q=campaign(scores,save,years,margin,cost,capacity)
a,b,c,d=st.columns(4)
a.metric('Flagged members',f'{len(flags):,}')
b.metric('Expected 90-day fund outflow',f'${scores.expected_outflow_90d.sum():,.0f}')
c.metric('Annualized fee revenue at risk',f'${scores.expected_annual_fee_revenue_at_risk.sum():,.0f}')
d.metric('Campaign net contribution',f'${q.expected_net_contribution.sum():,.0f}')
st.caption(f'{len(scores):,} held-out synthetic snapshots across July–December 2025; summed exposure is a pooled cohort total, not a single-date fund forecast. ROC AUC {m["roc_auc"]:.3f}. Assets belong to members; fee revenue is the administrator’s income.')
tabs=st.tabs(['Task 1 · Flag members','Task 2 · Money that may leave','Retention plan','Model evidence','Score member CSV'])
with tabs[0]:
    st.subheader('Members who might leave within 90 days')
    st.write(f'Probability ≥ {threshold:.2f} flags a member. The default maximizes validation F2 to weight recall more heavily than precision. A flag is a prioritization signal, not a certainty.')
    st.dataframe(flags[['member_id','snapshot_date','churn_probability_90d','churn_flag','account_balance','amount_if_member_leaves','expected_outflow_90d']],hide_index=True,width='stretch')
    st.download_button('Download flagged member list',flags.to_csv(index=False),'flagged_members.csv','text/csv')
    st.download_button('Download all predictions',scores.to_csv(index=False),'member_predictions.csv','text/csv')
with tabs[1]:
    st.subheader('How much would each departing member take?')
    st.write('Amount if member leaves = account balance × assumed transfer fraction. Expected outflow = churn probability × that amount. These are separate from the fee revenue lost when assets leave.')
    a,b,c=st.columns(3)
    a.metric('Flagged balances if all flagged members leave',f'${flags.amount_if_member_leaves.sum():,.0f}')
    b.metric('Expected outflow among flagged members',f'${flags.expected_outflow_90d.sum():,.0f}')
    c.metric('Expected outflow outside flagged group',f'${scores.loc[~scores.churn_flag,"expected_outflow_90d"].sum():,.0f}')
    st.dataframe(flags[['member_id','churn_probability_90d','amount_if_member_leaves','expected_outflow_90d','annual_fee_revenue_if_member_leaves','expected_annual_fee_revenue_at_risk']],hide_index=True,width='stretch')
    st.info('Annualized fee exposure = expected outflow × annual fee rate. It is the annual run-rate exposure after exit, not the revenue lost during the 90-day prediction window. No tax, withdrawal eligibility, investment return or fund-specific benefit rules are modeled.')
    st.subheader('What doing nothing means in this simulation')
    st.write(f'At the current transfer fraction, pooled model-estimated outflow is ${scores.expected_outflow_90d.sum():,.0f}. Separately, observed synthetic full-transfer outflow was ${m["observed_synthetic_outflow_90d"]:,.0f}. Historical labels are used only for evaluation, never as prediction inputs.')
with tabs[2]:
    st.subheader('Prioritize a retention pilot')
    st.write('Expected net contribution = probability × transferable balance × annual fee rate × save rate × fee years × margin − contact cost. Rank positive values within the flagged group; capacity is a ceiling, not a spending target.')
    a,b,c=st.columns(3)
    a.metric('Selected contacts',str(len(q)));b.metric('Expected member assets retained',f'${q.expected_assets_retained.sum():,.0f}');c.metric('Expected fee revenue preserved',f'${q.expected_fee_revenue_preserved.sum():,.0f}')
    st.dataframe(q,hide_index=True,width='stretch');st.download_button('Download retention priority queue',q.to_csv(index=False),'retention_priority.csv','text/csv')
    policies=[]
    for policy in ['value','risk','random']:
        c=campaign(scores,save,years,margin,cost,capacity,policy)
        policies.append({'Policy':policy,'Contacts':len(c),'Expected net contribution ($)':c.expected_net_contribution.sum()})
    st.dataframe(pd.DataFrame(policies),hide_index=True,width='stretch')
    st.caption('When capacity exceeds the eligible count, all policies select the same set. Reduce capacity to 50 to compare rankings.')
    st.line_chart(pd.DataFrame({'Assumed save rate':[i/20 for i in range(21)],'Expected net contribution ($)':[campaign(scores,i/20,years,margin,cost,capacity).expected_net_contribution.sum() for i in range(21)]}).set_index('Assumed save rate'))
    st.write('Suggested pilot: verify contact preferences and service issues, offer service support, and record outreach, offer cost and transfer outcome. Randomize outreach versus control within prespecified risk/value bands to measure incremental retained assets and fee contribution. Do not infer that changing an associated feature will prevent exit.')
    st.warning('Save rate is assumed, not learned. This project cannot prove it stops members leaving. Financial value is a scenario, not actual savings.')
with tabs[3]:
    st.subheader('Temporal holdout and model quality')
    st.write(f'Selected model: {m["winner"]}; validation Brier score selects the probability model; validation F2 selects the flag threshold. Base fitting, calibration, validation and test periods are separated by 90-day label-maturity gaps.')
    st.dataframe(pd.read_csv(ROOT/'outputs/validation_metrics.csv'),hide_index=True)
    st.json({k:m[k] for k in ['roc_auc','auc_ci95','average_precision','brier','precision','recall','f2','confusion_matrix']})
    st.caption('Validation comparison uses threshold 0.50; final test precision/recall use the selected threshold. Confusion matrix: actual rows [stay,leave], predicted columns [stay,leave].')
    st.image(str(ROOT/'outputs/evaluation.png'))
    st.dataframe(pd.read_csv(ROOT/'outputs/subgroup_audit.csv'),hide_index=True)
    st.caption('Age excluded from predictors. Audit is descriptive; synthetic group comparisons do not establish real-world fairness.')
with tabs[4]:
    st.subheader('Score a member file')
    st.write('Upload one snapshot per member using the example schema. Target/outcome columns are unnecessary. Balances and fees must be known at the snapshot date.')
    st.download_button('Download example input',(ROOT/'data/example_members.csv').read_bytes(),'example_members.csv','text/csv')
    uploaded=st.file_uploader('Member CSV',type='csv')
    if uploaded is not None:
        try:
            predicted=score_frame(pd.read_csv(uploaded),threshold,withdrawal)
            st.success(f'Scored {len(predicted):,} members; {int(predicted.churn_flag.sum()):,} flagged.')
            st.dataframe(predicted,hide_index=True,width='stretch')
            st.download_button('Download scored members',predicted.to_csv(index=False),'scored_members.csv','text/csv')
        except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
st.divider();st.caption('Designed by Nova Agentic · Portfolio by Joseph Ikogho · Local synthetic-data application')
