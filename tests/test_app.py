import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest

RUTA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_la_app_carga_y_predice():
    at = AppTest.from_file(RUTA, default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.metric) >= 2


def test_cambiar_a_pelicula_y_comparar_modelos():
    at = AppTest.from_file(RUTA, default_timeout=180).run()
    at.radio[0].set_value("Película").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) >= 1
