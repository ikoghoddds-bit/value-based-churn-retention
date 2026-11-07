from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def test_dashboard_and_zero_save_rate():
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not at.exception
    assert len(at.metric)>=4
    at.sidebar.slider[2].set_value(0.).run()
    assert not at.exception and at.metric[3].value=='$0'
