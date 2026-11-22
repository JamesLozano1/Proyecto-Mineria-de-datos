import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest


def test_la_app_carga_y_predice():
    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.metric) == 2
