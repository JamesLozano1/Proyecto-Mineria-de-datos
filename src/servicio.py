import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from src.lectorDatos import LectorDatos
from src.limpiadorDatos import LimpiadorDatos
from src.modeloArbol import ModeloArbol
from src.modeloKMeans import ModeloKMeans


class ServicioPrediccion:
    """
    Capa de despliegue: carga y limpia los datos, entrena con todos los títulos los modelos
    finales (árbol para explicar, Naive Bayes para predecir, K-means para segmentar)
    y ofrece funciones simples para la aplicación web.
    """

    K_GRUPOS = 10

    def __init__(self, carpetaCSV=None):
        base = carpetaCSV or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CSV")
        catalogo = LectorDatos(os.path.join(base, "netflix_titles.csv")).cargarDatos()
        limpiador = LimpiadorDatos(catalogo)
        limpiador.limpiarDatos()
        self.catalogo = limpiador.corregirYProcesar()
        self.valoraciones = LimpiadorDatos(LectorDatos(os.path.join(base, "netflix.csv")).cargarDatos()).limpiarDatos()

        self.arbol = ModeloArbol(maxProfundidad=4, umbral=85)
        self.X, self.y = self.arbol.prepararDatos(self.valoraciones, self.catalogo)
        self.arbol.entrenarCompleto()
        self.bayes = GaussianNB().fit(self.X, self.y)

        self.kmeans = ModeloKMeans()
        self.kmeans.prepararDatos(self.catalogo)
        self.kmeans.entrenar(self.K_GRUPOS)   # k elegido en el análisis (silhouette)
        self.perfilesGrupos = self.kmeans.perfiles()

    def generos(self):
        return [v[2:] for v in self.arbol.variables if v.startswith("g_")]

    def clasificaciones(self):
        """Clasificación (TV-MA, PG...) -> nivel numérico más frecuente en los datos."""
        v = self.valoraciones.dropna(subset=["rating", "rating_level"])
        return v.groupby("rating")["rating_level"].agg(lambda s: s.mode().iloc[0]).sort_index().to_dict()

    def construirFila(self, tipo, temporadas, minutos, anio, clasificacion, generos):
        niveles = self.clasificaciones()
        fila = {c: 0 for c in self.arbol.variables}
        fila["release_year"] = anio
        fila["rating_level"] = niveles.get(clasificacion, 0)
        fila["es_serie"] = int(tipo == "Serie")
        fila["temporadas"] = temporadas if tipo == "Serie" else 0
        fila["minutos"] = minutos if tipo == "Película" else 0
        fila["infantil"] = int(clasificacion in ModeloArbol.INFANTIL)
        for g in generos:
            fila["g_" + g] = 1
        return pd.Series(fila)

    def predecir(self, fila):
        x = fila[self.arbol.variables].to_frame().T.astype(float)
        probaBayes = float(self.bayes.predict_proba(x)[0][1])
        probaArbol = float(self.arbol.modelo.predict_proba(x)[0][1])
        return {"bayes": probaBayes, "arbol": probaArbol, "camino": self.arbol.explicarCamino(fila)}

    def buscarTitulo(self, texto):
        t = self.kmeans.titulos
        coincide = t[t["title"].str.contains(texto, case=False, regex=False)]
        return coincide[["title", "type", "release_year", "listed_in", "grupo"]].head(15)
