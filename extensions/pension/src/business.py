"""Money definitions: member assets are not the pension administrator's revenue."""
import numpy as np
import pandas as pd
FEATURES=['tenure_months','account_balance','monthly_contribution','months_since_contribution',
    'employment_status','portal_visits_90d','complaints_90d','transfer_inquiry_90d',
    'satisfaction_score','annual_fee_rate']
NUMERIC=[c for c in FEATURES if c!='employment_status']

def validate(d):
    required=['member_id','snapshot_date']+FEATURES
    missing=sorted(set(required)-set(d.columns))
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    d=d.copy()
    if d.empty or d.member_id.isna().any() or d.member_id.astype(str).str.strip().eq('').any() or not d.member_id.is_unique:
        raise ValueError('Supply nonempty data with unique, nonmissing member IDs (one snapshot per member).')
    if pd.to_datetime(d.snapshot_date,errors='coerce').isna().any():raise ValueError('snapshot_date must contain valid dates.')
    for c in NUMERIC:
        original=d[c]; d[c]=pd.to_numeric(original,errors='coerce')
        if c=='satisfaction_score':
            if (original.notna() & d[c].isna()).any():raise ValueError('Invalid satisfaction_score.')
            if not d[c].dropna().between(1,5).all():raise ValueError('satisfaction_score must be 1–5 or missing.')
        elif d[c].isna().any() or not np.isfinite(d[c]).all() or d[c].lt(0).any():
            raise ValueError(f'{c} must be finite, nonnegative and numeric.')
    if not d.annual_fee_rate.between(0,.05).all():raise ValueError('annual_fee_rate must be a decimal between 0 and 0.05.')
    if not d.transfer_inquiry_90d.isin([0,1]).all():raise ValueError('transfer_inquiry_90d must be 0 or 1.')
    if not d.employment_status.isin(['Employed','Self-employed','Not contributing']).all():raise ValueError('Unknown employment_status category.')
    return d

def attach_financials(d,probabilities,threshold=.3,withdrawal_fraction=1.):
    if not 0<=withdrawal_fraction<=1:raise ValueError('withdrawal_fraction must be between 0 and 1.')
    if not 0<=threshold<=1:raise ValueError('threshold must be between 0 and 1.')
    p=np.asarray(probabilities)
    if len(p)!=len(d) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities.')
    cols=['member_id','snapshot_date','account_balance','annual_fee_rate']
    s=d[cols].copy();s['churn_probability_90d']=p
    s['churn_flag']=p>=threshold
    s['amount_if_member_leaves']=s.account_balance*withdrawal_fraction
    s['expected_outflow_90d']=p*s.amount_if_member_leaves
    # Annualized run-rate exposure after exit, not revenue lost during the 90-day window.
    s['annual_fee_revenue_if_member_leaves']=s.amount_if_member_leaves*s.annual_fee_rate
    s['expected_annual_fee_revenue_at_risk']=p*s.annual_fee_revenue_if_member_leaves
    return s

def campaign(scores,save_rate=.25,fee_years=1.,margin=.7,cost=25.,capacity=200,strategy='value',flagged_only=True):
    if not (0<=save_rate<=1 and 0<=margin<=1 and fee_years>0 and cost>=0 and capacity>=0):raise ValueError('Invalid campaign assumptions.')
    d=scores.copy()
    d['expected_assets_retained']=d.expected_outflow_90d*save_rate
    d['expected_fee_revenue_preserved']=d.expected_annual_fee_revenue_at_risk*save_rate*fee_years
    d['expected_net_contribution']=d.expected_fee_revenue_preserved*margin-cost
    if flagged_only:d=d[d.churn_flag.astype(bool)]
    d=d[d.expected_net_contribution>0]
    if strategy=='value':d=d.sort_values(['expected_net_contribution','member_id'],ascending=[False,True])
    elif strategy=='risk':d=d.sort_values(['churn_probability_90d','member_id'],ascending=[False,True])
    elif strategy=='random':d=d.sample(frac=1,random_state=42)
    else:raise ValueError('Unknown strategy.')
    return d.head(int(capacity))
