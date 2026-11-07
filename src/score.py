"""python -m src.score input.csv output.csv"""
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def score_frame(d):
    bundle=joblib.load(ROOT/'outputs/model.joblib') # Load trusted local artifact only.
    required=bundle['features']+['customerID']
    missing=sorted(set(required)-set(d.columns))
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    if d.empty or d.customerID.isna().any() or d.customerID.astype(str).str.strip().eq('').any() or not d.customerID.is_unique:
        raise ValueError('Supply nonempty data with unique, nonmissing customerID values.')
    d=d.copy()
    for col in ['tenure','MonthlyCharges','TotalCharges']:
        original=d[col]
        d[col]=pd.to_numeric(original,errors='coerce')
        if col=='TotalCharges':
            blank=original.isna()|original.astype(str).str.strip().eq('')
            if ((~blank)&d[col].isna()).any() or not np.isfinite(d[col].dropna()).all() or d[col].dropna().lt(0).any():
                raise ValueError('TotalCharges must be nonnegative numeric or blank.')
    if not np.isfinite(d[['tenure','MonthlyCharges']].to_numpy()).all() or d[['tenure','MonthlyCharges']].isna().any().any() or (d[['tenure','MonthlyCharges']]<0).any().any():
        raise ValueError('tenure and MonthlyCharges must be nonnegative numeric values.')
    # Reject typos rather than silently assigning unseen categories zero weight.
    reference=pd.read_csv(ROOT/'data/telco.csv')
    for col in bundle['features']:
        if reference[col].dtype=='object' and col!='TotalCharges':
            if not d[col].isin(reference[col].unique()).all():
                raise ValueError(f'Unknown or missing category in {col}. Match the example CSV.')
    result=d[['customerID','MonthlyCharges']].copy()
    result['probability']=bundle['model'].predict_proba(d[bundle['features']])[:,1]
    return result

if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('Usage: python -m src.score input.csv output.csv')
    score_frame(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2],index=False)
