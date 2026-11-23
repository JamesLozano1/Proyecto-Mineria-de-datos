import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

    # ---------- datos para las gráficas de la aplicación ----------

    NOMBRES = {"temporadas": "Temporadas", "minutos": "Minutos", "release_year": "Año de lanzamiento",
               "rating_level": "Nivel de clasificación", "es_serie": "Es serie", "infantil": "Contenido infantil"}

    def nombreVariable(self, v):
        return self.NOMBRES.get(v, "Género: " + v[2:] if v.startswith("g_") else v)

    def indicadores(self):
        c = self.catalogo.drop_duplicates("title")
        v = self.valoraciones.dropna(subset=["user_rating_score"])
        return {"titulos": len(c), "peliculas": (c["type"] == "Movie").mean(), "series": (c["type"] == "TV Show").mean(),
                "valoracion": v["user_rating_score"].mean(), "valorados": len(v)}

    def importanciaArbol(self):
        imp = pd.Series(self.arbol.modelo.feature_importances_, index=self.arbol.variables)
        imp = imp[imp > 0].sort_values(ascending=False).head(8)
        imp.index = [self.nombreVariable(i) for i in imp.index]
        return imp

    def sensibilidad(self, tipo, anio, clasificacion, generos, temporadas, minutos):
        """Probabilidad de los dos modelos al variar las temporadas (series) o los minutos (películas)."""
        if tipo == "Serie":
            valores, etiqueta = list(range(1, 18)), "Temporadas"
        else:
            valores, etiqueta = list(range(60, 201, 10)), "Minutos"
        filas = []
        for x in valores:
            fila = self.construirFila(tipo, x if tipo == "Serie" else temporadas, x if tipo != "Serie" else minutos,
                                      anio, clasificacion, generos)
            r = self.predecir(fila)
            filas.append({"x": x, "modelo": "Naive Bayes", "probabilidad": r["bayes"]})
            filas.append({"x": x, "modelo": "Árbol de decisión", "probabilidad": r["arbol"]})
        return pd.DataFrame(filas), etiqueta

    def valoracionPorGenero(self, minimo=3):
        v = self.valoraciones.dropna(subset=["user_rating_score"])
        cat = self.catalogo.drop_duplicates("title")[["title", "listed_in"]]
        m = v.merge(cat, on="title")
        m = m.assign(genero=m["listed_in"].str.split(", ")).explode("genero")
        m["genero"] = m["genero"].str.strip()
        t = m.groupby("genero")["user_rating_score"].agg(titulos="count", promedio="mean", mediana="median").reset_index()
        return t[t["titulos"] >= minimo].sort_values("promedio", ascending=False)

    def valoracionAudiencia(self):
        v = self.valoraciones.dropna(subset=["user_rating_score"]).copy()
        adultos = ["TV-14", "TV-MA", "R", "PG-13", "NR"]
        v["audiencia"] = np.where(v["rating"].isin(ModeloArbol.INFANTIL), "Infantil/Familiar",
                                  np.where(v["rating"].isin(adultos), "Adolescente/Adulto", "Otra (PG, TV-PG, UR)"))
        return v[["audiencia", "user_rating_score"]]

    def coordenadasGrupos(self, muestra=3000):
        if not hasattr(self, "_coordenadas"):
            pts = PCA(n_components=2, random_state=42).fit_transform(self.kmeans.Xs)
            t = self.kmeans.titulos[["title", "grupo"]].copy()
            t["x"], t["y"] = pts[:, 0], pts[:, 1]
            t["nombre"] = t["grupo"].map(self.perfilesGrupos["nombre"])
            t["grupo"] = "Grupo " + t["grupo"].astype(str)
            self._coordenadas = t.sample(min(muestra, len(t)), random_state=42)
        return self._coordenadas

    def centroides(self):
        c = pd.DataFrame(self.kmeans.modelo.cluster_centers_, columns=[self.nombreVariable(v).replace("Género: ", "") for v in self.kmeans.variables.copy()])
        c["grupo"] = ["Grupo %d" % (i + 1) for i in range(len(c))]
        return c.melt(id_vars="grupo", var_name="variable", value_name="valor")
