"""Local recruiter demo. Run: streamlit run app.py"""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.business import prioritize,campaign
from src.score import score_frame
ROOT=Path(__file__).parent
st.set_page_config(page_title='Value-based churn | Joseph Ikogho',page_icon='◈',layout='wide')
st.markdown('''<style>.stApp{background:#f5f8fa}h1,h2,h3{color:#123b46}div[data-testid="stMetric"]{background:white;padding:18px;border-radius:12px;border-top:3px solid #07828b}</style>''',unsafe_allow_html=True)
st.caption('JOSEPH IKOGHO · DATA SCIENCE PORTFOLIO / 01')
st.title('Value-Based Customer Churn Prediction & Prioritized Retention List')
st.write('Identify potential churners, quantify monthly revenue exposure, and decide whom to call first.')
if not (ROOT/'outputs/metrics.json').exists():st.error('Run python -m src.train first.');st.stop()
m=json.loads((ROOT/'outputs/metrics.json').read_text());evaluation=pd.read_csv(ROOT/'outputs/test_scores.csv')
# Targets are stripped before constructing action lists: no hindsight in outreach.
d=evaluation.drop(columns=['actual_churn','gender','SeniorCitizen'])
with st.sidebar:
    st.header('Screening and economics')
    threshold=st.slider('Churn flag threshold',0.,1.,float(m['threshold']),.01)
    months=st.slider('Retained revenue months',1,12,6)
    success=st.slider('Would-be churners saved',0.,1.,.25,.05)
    margin=st.slider('Contribution margin',0.,1.,.70,.05)
    capacity=st.slider('Contact capacity',0,len(d),200)
    cost=st.number_input('Uniform campaign all-in cost ($)',0.,300.,20.,5.)
    st.header('Illustrative support tiers')
    outreach=st.number_input('Outreach cost per customer ($)',0.,100.,5.,1.)
    promo=st.number_input('Promo-code cost ($)',0.,200.,10.,1.)
    advisor=st.number_input('Dedicated-advisor cost ($)',float(promo),500.,max(35.,float(promo)),1.)
    st.caption('Treatment effectiveness is assumed, not measured. Tier labels illustrate affordability; they are not personalized causal recommendations.')
ranked=prioritize(d,threshold,months,success,margin,outreach,promo,advisor)
q=campaign(d,success,months,margin,cost,capacity)
a,b,c,e=st.columns(4)
a.metric('Held-out recall',f'{m["recall"]:.1%}')
b.metric('Expected monthly revenue loss',f'${ranked.expected_monthly_loss.sum():,.0f}')
c.metric('Flagged customers',str(int(ranked.churn_flag.sum())))
e.metric('Uniform campaign net contribution',f'${q.expected_net_value.sum():,.0f}')
st.caption(f'{len(d):,} held-out IBM Telco sample customers · USD · All dollar results are modeled exposure/scenarios, not realized savings. Recall above uses the frozen validation-selected threshold, not the interactive slider.')
tabs=st.tabs(['Prioritized call list','Incentive limits and ROI','Recall and error costs','Model evidence','Score customer CSV'])
with tabs[0]:
    st.subheader('Whom should the retention team call first?')
    st.latex(r'\mathrm{Expected\ monthly\ loss}_i=P(\mathrm{churn})_i\times\mathrm{MonthlyValue}_i')
    st.write('The list ranks every customer by this exact score. The risk flag is shown separately: filtering only flags can omit a high-value account with a lower probability.')
    show_flags=st.checkbox('Display flagged customers only',False)
    view=ranked[ranked.churn_flag] if show_flags else ranked
    cols=['priority_rank','customerID','probability','churn_flag','monthly_customer_value','expected_monthly_loss','max_total_retention_spend','max_incentive_after_outreach','suggested_support_tier','planned_total_cost','expected_net_after_suggested_tier']
    st.dataframe(view[cols],hide_index=True,width='stretch')
    st.download_button('Download prioritized call list',view.to_csv(index=False),'prioritized_call_list.csv','text/csv')
    st.caption('Capacity and profitability further constrain the separate campaign queue. The core ranking itself remains probability × monthly value.')
with tabs[1]:
    st.subheader('How much can retention cost before value disappears?')
    st.write('Break-even total spend = probability × monthly value × retained months × save rate × margin. Maximum incentive = max(0, break-even spend − outreach cost). Spend below break-even to preserve positive expected contribution.')
    st.write('Illustrative tier rule: advisor if its cost fits strictly below the cap, otherwise promo code, otherwise outreach only, otherwise no paid outreach. This shows affordable support intensity; it does not establish that the more expensive tier is more effective.')
    st.subheader('Separate uniform-cost campaign scenario')
    st.caption('This campaign applies the sidebar all-in cost to every contact; it does not also charge the tier costs. Both use the same assumed save rate for transparent comparisons.')
    a,b,c=st.columns(3);a.metric('Contacts',str(len(q)));b.metric('Expected saved revenue',f'${q.expected_saved_revenue.sum():,.0f}');c.metric('Expected net contribution',f'${q.expected_net_value.sum():,.0f}')
    st.dataframe(q,hide_index=True,width='stretch');st.download_button('Download uniform-cost campaign queue',q.to_csv(index=False),'retention_queue.csv','text/csv')
    rows=[]
    for policy in ['value','risk','random']:
        v=campaign(d,success,months,margin,cost,capacity,policy)
        rows.append({'Policy':policy,'Contacts':len(v),'Expected net contribution':v.expected_net_value.sum()})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.line_chart(pd.DataFrame({'Save rate':[i/20 for i in range(21)],'Expected net contribution':[campaign(d,i/20,months,margin,cost,capacity).expected_net_value.sum() for i in range(21)]}).set_index('Save rate'))
    st.warning('Six retained months is a constant-revenue valuation assumption, not a six-month churn forecast or customer lifetime value. No actual treatment outcomes are supplied.')
with tabs[2]:
    st.subheader('Missing a churner versus calling a customer who stays')
    st.write(f'The default threshold {m["threshold"]:.2f} is the highest threshold attaining at least 85% validation recall. This prioritizes finding churners while limiting unnecessary calls; pure recall alone would flag everyone.')
    st.dataframe(pd.DataFrame([{'Policy':'Recall-first','Recall':m['recall'],'Precision':m['precision'],'Threshold':m['threshold']},{'Policy':'Default threshold','Recall':m['test_at_threshold_0_5']['recall'],'Precision':m['test_at_threshold_0_5']['precision'],'Threshold':.5}]),hide_index=True)
    st.write('Illustrative screening penalties: $20 per false positive, $100 per false negative (5:1). These are diagnostic assumptions, separate from retention ROI, not measured company costs. Sensitivity examines ratios 1:1, 3:1, 5:1 and 10:1.')
    st.dataframe(pd.read_csv(ROOT/'outputs/error_cost_sensitivity.csv'),hide_index=True,width='stretch')
    st.caption('Cost-minimizing comparison thresholds are selected on validation only; costs shown here are evaluated on held-out labels. A recall target need not minimize every assumed cost ratio.')
with tabs[3]:
    st.subheader('Simple model, business-first evaluation')
    st.write('A prior baseline and one logistic regression. Five-fold sigmoid calibration with preprocessing inside training folds; stratified 60/20/20 holdout. No complex model search. No timestamps are available for temporal validation.')
    st.json({k:m[k] for k in ['roc_auc','auc_ci95','pr_auc','brier','recall','precision','f2','confusion_matrix']})
    st.image(str(ROOT/'outputs/evaluation.png'))
    st.dataframe(pd.read_csv(ROOT/'outputs/subgroup_audit.csv'),hide_index=True)
    st.caption('Confusion matrix: actual rows [stay,leave], predicted columns [stay,leave]. Demographic exclusion does not eliminate proxy bias. Feature importance describes associations, not causal remedies.')
with tabs[4]:
    st.subheader('Score a customer CSV')
    st.write('Use the example schema; target Churn and demographics are unnecessary. IDs must be unique; charges and tenure must be finite and nonnegative.')
    st.download_button('Download example CSV',(ROOT/'data/example_customers.csv').read_bytes(),'example_customers.csv','text/csv')
    upload=st.file_uploader('Customer CSV',type='csv')
    if upload is not None:
        try:
            v=score_frame(pd.read_csv(upload))
            result=prioritize(v,threshold,months,success,margin,outreach,promo,advisor)
            st.dataframe(result,hide_index=True,width='stretch')
            st.download_button('Download scored call list',result.to_csv(index=False),'scored_call_list.csv','text/csv')
        except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
st.divider();st.caption('Designed by Nova Agentic · Joseph Ikogho · Public IBM sample benchmark · Synthetic pension extension included separately')
