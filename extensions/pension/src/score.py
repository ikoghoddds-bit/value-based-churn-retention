"""python -m src.score data/example_members.csv outputs/new_member_scores.csv"""
from pathlib import Path
import sys
import pandas as pd
import joblib
from src.business import validate,attach_financials
ROOT=Path(__file__).resolve().parents[1]

def score_frame(d,threshold=None,withdrawal_fraction=1.):
    d=validate(d);b=joblib.load(ROOT/'outputs/model.joblib') # Trusted local artifact only.
    p=b['model'].predict_proba(d[b['features']])[:,1]
    return attach_financials(d,p,b['threshold'] if threshold is None else threshold,withdrawal_fraction)

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: python -m src.score input.csv output.csv')
    score_frame(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2],index=False)
