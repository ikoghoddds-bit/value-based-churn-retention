import json
from pathlib import Path
import pandas as pd
import pytest
from src.train import campaign
from src.score import score_frame
ROOT=Path(__file__).resolve().parents[1]

def test_economics_and_capacity():
    d=pd.DataFrame({'customerID':['a','b'],'probability':[.8,.2],'MonthlyCharges':[100.,50.]})
    q=campaign(d,success=.25,months=6,margin=.7,cost=20,capacity=1)
    assert q.customerID.tolist()==['a']
    assert q.expected_net_value.iloc[0]==pytest.approx(64)
    assert q.expected_saved_revenue.iloc[0]==pytest.approx(120)
    assert campaign(d,success=0).empty
    assert campaign(d,capacity=0).empty
    with pytest.raises(ValueError): campaign(d,success=2)

def test_split_integrity():
    s=pd.read_csv(ROOT/'outputs/splits.csv')
    assert s.customerID.is_unique
    assert len(s)==7043
    assert set(s.split)=={'train','validation','test'}

def test_scoring_and_schema():
    d=pd.read_csv(ROOT/'data/telco.csv').head(8)
    s=score_frame(d.drop(columns='Churn'))
    assert len(s)==8 and s.probability.between(0,1).all()
    with pytest.raises(ValueError): score_frame(d.drop(columns='Contract'))
    d.loc[0,'MonthlyCharges']=-1
    with pytest.raises(ValueError): score_frame(d)

def test_saved_predictions_match():
    source=pd.read_csv(ROOT/'data/telco.csv')
    saved=pd.read_csv(ROOT/'outputs/test_scores.csv')
    source=source.set_index('customerID').loc[saved.customerID].reset_index()
    rescored=score_frame(source)
    assert abs(rescored.probability.to_numpy()-saved.probability.to_numpy()).max()<1e-10

def test_exact_monthly_ranking_and_affordable_tiers():
    from src.business import prioritize
    d=pd.DataFrame({'customerID':['high-risk','high-value'],'probability':[.9,.5],'MonthlyCharges':[20.,100.]})
    q=prioritize(d,.7)
    assert q.customerID.tolist()==['high-value','high-risk']
    assert q.expected_monthly_loss.tolist()==[50.,18.]
    assert q.churn_flag.tolist()==[False,True]
    assert q.max_total_retention_spend.iloc[0]==pytest.approx(52.5)
    assert q.max_incentive_after_outreach.iloc[0]==pytest.approx(47.5)
    assert q.suggested_support_tier.iloc[0]=='Dedicated advisor'
    assert q.expected_net_after_suggested_tier.iloc[0]==pytest.approx(12.5)
    assert (q.planned_total_cost<q.max_total_retention_spend).all()
    assert prioritize(d,.7,save_rate=0).max_incentive_after_outreach.eq(0).all()
    assert prioritize(d,.7,save_rate=0).suggested_support_tier.eq('No paid outreach').all()

def test_recall_threshold_and_outreach_exports():
    m=json.loads((ROOT/'outputs/metrics.json').read_text())
    tradeoffs=pd.read_csv(ROOT/'outputs/validation_threshold_tradeoffs.csv')
    eligible=tradeoffs[tradeoffs.recall>=.85]
    assert m['threshold']==pytest.approx(eligible.threshold.max())
    assert m['validation_recall']>=.85
    q=pd.read_csv(ROOT/'outputs/prioritized_call_list.csv')
    assert 'actual_churn' not in q.columns
    assert q.expected_monthly_loss.is_monotonic_decreasing
    assert q.priority_rank.tolist()==list(range(1,len(q)+1))
    assert (q.expected_monthly_loss-q.probability*q.MonthlyCharges).abs().max()<1e-10


def test_nonfinite_scoring_rejected():
    import numpy as np
    d=pd.read_csv(ROOT/'data/telco.csv').head(2)
    for col,value in [('MonthlyCharges',np.inf),('TotalCharges','oops'),('customerID','')]:
        bad=d.copy();bad[col]=bad[col].astype(object);bad.loc[0,col]=value
        with pytest.raises(ValueError):score_frame(bad)
