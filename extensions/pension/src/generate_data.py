"""Explicitly synthetic pension-member snapshots. No real people or fund records.
The data-generating process is illustrative, not calibrated to a pension industry.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SEED=2026

def generate():
    rng=np.random.default_rng(SEED)
    dates=np.repeat(pd.date_range('2023-01-31','2025-12-31',freq='ME'),400)
    n=len(dates); tenure=rng.integers(1,241,n)
    inactive=rng.choice([0,1,2,3,6,9,12],n,p=[.48,.16,.12,.08,.07,.05,.04])
    employment=rng.choice(['Employed','Self-employed','Not contributing'],n,p=[.7,.2,.1])
    balance=np.clip(rng.lognormal(10.4,.95,n)*(1+tenure/250),2000,1500000).round(2)
    contribution=np.where(employment=='Not contributing',0,np.clip(rng.lognormal(6.1,.6,n),25,3500)).round(2)
    visits=rng.poisson(3,n); complaints=np.clip(rng.poisson(.5,n),0,6)
    inquiries=rng.binomial(1,.16,n); satisfaction=rng.choice([1,2,3,4,5],n,p=[.08,.12,.25,.32,.23]).astype(float)
    fee=rng.choice([.003,.005,.0075,.01],n,p=[.2,.4,.3,.1])
    # Only pre-snapshot features drive a simulated voluntary full-transfer event.
    latent=(-2.5+.16*inactive+.4*complaints+1.35*inquiries-.16*visits
            +.35*(3-satisfaction)+.55*(employment=='Not contributing')
            +.55*(fee==.01)-.002*tenure+.1*(pd.DatetimeIndex(dates).year-2023))
    probability=1/(1+np.exp(-latent))
    churn=rng.binomial(1,probability)
    d=pd.DataFrame({'member_id':[f'SYN-{i:06d}' for i in range(1,n+1)],'snapshot_date':dates,
        'age':rng.integers(25,61,n),'tenure_months':tenure,'account_balance':balance,
        'monthly_contribution':contribution,'months_since_contribution':inactive,
        'employment_status':employment,'portal_visits_90d':visits,'complaints_90d':complaints,
        'transfer_inquiry_90d':inquiries,'satisfaction_score':satisfaction,'annual_fee_rate':fee,
        'churn_90d':churn})
    # Missing survey responses represent a realistic input-quality issue.
    d.loc[rng.choice(n,int(n*.06),replace=False),'satisfaction_score']=np.nan
    d['outcome_window_end']=d.snapshot_date+pd.Timedelta(days=90)
    d['actual_outflow_90d']=d.churn_90d*d.account_balance
    # No latent probability or post-outcome column enters the feature matrix.
    ROOT.joinpath('data').mkdir(exist_ok=True)
    d.to_csv(ROOT/'data/synthetic_members.csv',index=False)
    print(f'Generated {n:,} explicitly synthetic unique member snapshots.')
    return d

if __name__=='__main__': generate()
