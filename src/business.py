"""Independent decision rules. Money fields are scenarios, never realized savings."""
import numpy as np
import pandas as pd

def prioritize(frame,threshold,months=6,save_rate=.25,margin=.7,outreach_cost=5.,promo_cost=10.,advisor_cost=35.):
    """The required ranking is exactly P(churn) × monthly customer value.

    The screening flag and the campaign budget are separate decisions. We retain
    all customers here so a low-probability high-value account is not discarded.
    An affordability tier is illustrative, not an estimated treatment effect.
    """
    values=[threshold,months,save_rate,margin,outreach_cost,promo_cost,advisor_cost]
    if not all(np.isfinite(v) for v in values) or not (0<=threshold<=1 and months>0 and 0<=save_rate<=1 and 0<=margin<=1 and 0<=outreach_cost and 0<=promo_cost<=advisor_cost):
        raise ValueError('Invalid financial assumptions or threshold.')
    d=frame.copy()
    p=d.probability.to_numpy();v=d.MonthlyCharges.to_numpy()
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isfinite(v).all() or (v<0).any():raise ValueError('Probabilities and monthly values must be valid and finite.')
    d['churn_flag']=p>=threshold
    d['monthly_customer_value']=d.MonthlyCharges
    d['expected_monthly_loss']=p*v
    d['revenue_at_risk_scenario']=d.expected_monthly_loss*months
    d['expected_saved_revenue']=d.revenue_at_risk_scenario*save_rate
    d['max_total_retention_spend']=d.expected_saved_revenue*margin
    d['max_incentive_after_outreach']=(d.max_total_retention_spend-outreach_cost).clip(lower=0)
    # Spend strictly below break-even: a cap exactly equal to a tier has zero
    # modeled contribution, so it falls to a cheaper tier.
    cap=d.max_incentive_after_outreach
    d['suggested_support_tier']=np.select([cap>advisor_cost,cap>promo_cost,cap>0],['Dedicated advisor','Promo code','Outreach only'],default='No paid outreach')
    d['planned_incentive_cost']=np.select([cap>advisor_cost,cap>promo_cost],[advisor_cost,promo_cost],default=0.)
    d['planned_total_cost']=np.where(cap>0,d.planned_incentive_cost+outreach_cost,0.)
    d['expected_net_after_suggested_tier']=np.where(cap>0,d.max_total_retention_spend-d.planned_total_cost,0.)
    d=d.sort_values(['expected_monthly_loss','customerID'],ascending=[False,True]).reset_index(drop=True)
    d.insert(0,'priority_rank',np.arange(1,len(d)+1))
    return d

def campaign(frame,success=.25,months=6,margin=.7,cost=20.,capacity=200,strategy='value'):
    """Uniform all-in $20 campaign; kept separate from affordability-tier display.

    Save rate is the assumed fraction of otherwise churners prevented from
    leaving. Costs apply to every contacted customer, even false positives.
    """
    args=[success,months,margin,cost,capacity]
    if not all(np.isfinite(v) for v in args) or not (0<=success<=1 and months>0 and 0<=margin<=1 and cost>=0 and capacity>=0):raise ValueError('Invalid campaign assumptions.')
    d=frame.copy();d['expected_monthly_loss']=d.probability*d.MonthlyCharges
    d['revenue_at_risk']=d.expected_monthly_loss*months
    d['expected_saved_revenue']=d.revenue_at_risk*success
    d['expected_net_value']=d.expected_saved_revenue*margin-cost
    if strategy=='value':d=d.sort_values(['expected_monthly_loss','customerID'],ascending=[False,True])
    elif strategy=='risk':d=d.sort_values(['probability','customerID'],ascending=[False,True])
    elif strategy=='random':d=d.sample(frac=1,random_state=42)
    else:raise ValueError('Unknown strategy.')
    return d[d.expected_net_value>0].head(int(capacity))
