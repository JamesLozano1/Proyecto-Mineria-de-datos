import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from src.servicio import ServicioPrediccion


@pytest.fixture(scope="module")
def servicio():
    return ServicioPrediccion()


def test_fechas_recuperadas(servicio):
    # 88 fechas traían un espacio al inicio; con strip solo quedan las 10 realmente vacías
    assert servicio.catalogo["date_added"].isna().sum() <= 10


def test_duplicados_eliminados(servicio):
    assert len(servicio.valoraciones) == 500


def test_arbol_profundidad_maxima_4(servicio):
    assert servicio.arbol.modelo.get_depth() <= 4
    assert len(servicio.X) == 137


def test_kmeans_asigna_todos_los_titulos(servicio):
    assert servicio.kmeans.k == 10
    assert servicio.kmeans.titulos["grupo"].between(1, 10).all()


def test_prediccion_devuelve_probabilidades_y_camino(servicio):
    fila = servicio.construirFila("Serie", 8, 0, 2015, "TV-14", ["TV Dramas"])
    r = servicio.predecir(fila)
    assert 0 <= r["bayes"] <= 1 and 0 <= r["arbol"] <= 1
    assert len(r["camino"]) >= 1
