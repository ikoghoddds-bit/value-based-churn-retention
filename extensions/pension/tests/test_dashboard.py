from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def test_dashboard_financial_controls():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].value=='807'
    at.sidebar.slider[2].set_value(0.).run()
    assert not at.exception and at.metric[3].value=='$0'
    at.sidebar.slider[1].set_value(0.).run()
    assert not at.exception and at.metric[1].value=='$0' and at.metric[2].value=='$0'
