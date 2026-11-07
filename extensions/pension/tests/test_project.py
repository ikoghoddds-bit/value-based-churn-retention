from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.business import FEATURES,validate,attach_financials,campaign
from src.score import score_frame
ROOT=Path(__file__).resolve().parents[1]

def test_member_assets_are_not_revenue():
    d=pd.DataFrame({'member_id':['a','b'],'snapshot_date':['2025-01-01']*2,'account_balance':[100000.,50000.],'annual_fee_rate':[.005,.005]})
    s=attach_financials(d,[.8,.2],.3)
    assert s.amount_if_member_leaves.tolist()==[100000.,50000.]
    assert s.expected_outflow_90d.iloc[0]==pytest.approx(80000.)
    assert s.expected_annual_fee_revenue_at_risk.iloc[0]==pytest.approx(400.)
    q=campaign(s,save_rate=.25,fee_years=1,margin=.7,cost=25,capacity=1)
    assert q.member_id.tolist()==['a'] and q.expected_net_contribution.iloc[0]==pytest.approx(45.)
    assert q.expected_assets_retained.iloc[0]==pytest.approx(20000.)
    assert campaign(s,save_rate=0).empty and campaign(s,capacity=0).empty
    assert attach_financials(d,[.8,.2],withdrawal_fraction=0).expected_outflow_90d.sum()==0
    with pytest.raises(ValueError):attach_financials(d,[np.inf,.2])
    with pytest.raises(ValueError):campaign(s,save_rate=2)

def test_temporal_boundaries_and_leakage():
    s=pd.read_csv(ROOT/'outputs/splits.csv');assert s.member_id.is_unique
    assert set(s.split)=={'train','validation','test','embargo'}
    for before,after in [('train','validation'),('validation','test')]:
        assert pd.to_datetime(s.loc[s.split==before,'outcome_window_end']).max()<pd.to_datetime(s.loc[s.split==after,'snapshot_date']).min()
    assert not set(FEATURES)&{'churn_90d','actual_outflow_90d','outcome_window_end','age','member_id'}
    d=pd.read_csv(ROOT/'data/synthetic_members.csv')
    fit=d[pd.to_datetime(d.snapshot_date)<='2023-12-31']
    cal=d[pd.to_datetime(d.snapshot_date).between('2024-04-01','2024-06-30')]
    assert pd.to_datetime(fit.outcome_window_end).max()<pd.to_datetime(cal.snapshot_date).min()

def test_schema_and_prediction_consistency():
    original=pd.read_csv(ROOT/'data/synthetic_members.csv');saved=pd.read_csv(ROOT/'outputs/member_predictions.csv')
    d=original.set_index('member_id').loc[saved.member_id].reset_index()
    actual=score_frame(d)
    np.testing.assert_allclose(actual.churn_probability_90d,saved.churn_probability_90d,atol=1e-12)
    assert actual.churn_probability_90d.between(0,1).all()
    for column,value in [('account_balance',-1),('annual_fee_rate',1),('monthly_contribution',np.inf),('employment_status','typo')]:
        bad=d.head(2).copy();bad.loc[0,column]=value
        with pytest.raises(ValueError):validate(bad)
    with pytest.raises(ValueError):validate(d.drop(columns='account_balance'))
    with pytest.raises(ValueError):validate(pd.concat([d.head(1)]*2))

def test_model_round_trip_and_cost_eligibility():
    scores=pd.read_csv(ROOT/'outputs/member_predictions.csv')
    q=campaign(scores)
    assert len(q)<=200 and q.churn_flag.all() and q.expected_net_contribution.gt(0).all()
    assert q.expected_net_contribution.is_monotonic_decreasing
    for strategy in ['risk','random']:
        alternative=campaign(scores,strategy=strategy)
        assert q.expected_net_contribution.sum()>=alternative.expected_net_contribution.sum()-1e-9
