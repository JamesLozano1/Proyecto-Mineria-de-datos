import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class ModeloKMeans:
    """
    Segmenta el catálogo de Netflix en grupos de títulos parecidos (K-means).
    Mide distancias, por eso las variables se escalan con StandardScaler
    (el año, ~2.000, dominaría sobre las variables 0/1).
    """

    INFANTIL = ["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"]
    ADULTO = ["TV-MA", "R", "NC-17"]

    def __init__(self, semilla=42, topGeneros=12):
        self.semilla = semilla
        self.topGeneros = topGeneros
        self.modelo = None
        self.k = None

    def prepararDatos(self, catalogo):
        t = catalogo.drop_duplicates("title").copy()
        t["es_serie"] = (t["type"] == "TV Show").astype(int)
        t["minutos"] = np.where(t["es_serie"] == 0, t["durationValue"], 0)
        t["temporadas"] = np.where(t["es_serie"] == 1, t["durationValue"], 0)
        t["infantil"] = t["rating"].isin(self.INFANTIL).astype(int)
        t["adulto"] = t["rating"].isin(self.ADULTO).astype(int)

        generos = t.assign(g=t["listed_in"].str.split(", ")).explode("g")
        generos["g"] = generos["g"].str.strip()
        self.generos = generos["g"].value_counts().head(self.topGeneros).index.tolist()
        for g in self.generos:
            t["g_" + g] = t["listed_in"].str.contains(g, regex=False).astype(int)

        self.variables = ["release_year", "es_serie", "minutos", "temporadas", "infantil", "adulto"] \
            + ["g_" + g for g in self.generos]
        self.titulos = t.reset_index(drop=True)
        self.X = self.titulos[self.variables].fillna(0)
        self.escalador = StandardScaler().fit(self.X)
        self.Xs = self.escalador.transform(self.X)
        return self.X

    def evaluarK(self, kMin=2, kMax=12):
        filas = []
        for k in range(kMin, kMax + 1):
            m = KMeans(k, n_init=10, random_state=self.semilla).fit(self.Xs)
            filas.append({"k": k, "inercia": m.inertia_,
                          "silhouette": silhouette_score(self.Xs, m.labels_),
                          "calinski_harabasz": calinski_harabasz_score(self.Xs, m.labels_),
                          "davies_bouldin": davies_bouldin_score(self.Xs, m.labels_)})
        self.evaluacion = pd.DataFrame(filas).set_index("k")
        return self.evaluacion

    def elegirK(self, tolerancia=0.01):
        """
        El silhouette sube y luego se estabiliza, así que se toma el menor k cuyo
        silhouette esté a `tolerancia` o menos del máximo (evita grupos de más).
        """
        sil = self.evaluacion["silhouette"]
        return int(sil[sil >= sil.max() - tolerancia].index.min())

    def entrenar(self, k=None):
        if k is None:
            if not hasattr(self, "evaluacion"):
                self.evaluarK()
            k = self.elegirK()
        self.k = k
        self.modelo = KMeans(self.k, n_init=10, random_state=self.semilla).fit(self.Xs)
        self.titulos["grupo"] = self.modelo.labels_ + 1
        return self.modelo

    def perfiles(self):
        t = self.titulos
        filas = []
        for g, d in t.groupby("grupo"):
            generos = d["listed_in"].str.split(", ").explode().str.strip().value_counts(normalize=True).head(3)
            filas.append({
                "grupo": g,
                "titulos": len(d),
                "porcentaje": len(d) / len(t) * 100,
                "pct_series": d["es_serie"].mean() * 100,
                "anio_medio": d["release_year"].mean(),
                "minutos_peliculas": d.loc[d["es_serie"] == 0, "minutos"].median() if (d["es_serie"] == 0).any() else np.nan,
                "temporadas_series": d.loc[d["es_serie"] == 1, "temporadas"].mean() if (d["es_serie"] == 1).any() else np.nan,
                "pct_infantil": d["infantil"].mean() * 100,
                "pct_adulto": d["adulto"].mean() * 100,
                "generos_principales": ", ".join(generos.index),
            })
        tabla = pd.DataFrame(filas).set_index("grupo")
        tabla["nombre"] = [self._nombre(r) for _, r in tabla.iterrows()]
        return tabla

    @staticmethod
    def _nombre(r):
        tipo = "Series" if r["pct_series"] >= 50 else "Películas"
        if r["pct_infantil"] >= 40:
            audiencia = "infantil/familiar"
        elif r["pct_adulto"] >= 50:
            audiencia = "adulto"
        else:
            audiencia = "general"
        return f"{tipo} · {r['generos_principales'].split(', ')[0]} · {audiencia}"

    def valoracionPorGrupo(self, valoraciones, minimo=5):
        """Valoración promedio por grupo (solo para los títulos que tienen puntuación y cruzan con el catálogo)."""
        v = valoraciones.dropna(subset=["user_rating_score"])
        m = v.merge(self.titulos[["title", "grupo"]].drop_duplicates("title"), on="title", how="inner")
        tabla = m.groupby("grupo")["user_rating_score"].agg(titulos="count", promedio="mean", mediana="median")
        tabla["confiable"] = tabla["titulos"] >= minimo
        return tabla

    def _guardar(self, ruta, mostrar):
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        plt.show() if mostrar else plt.close()

    def guardarGraficaK(self, ruta, mostrar=False):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].plot(self.evaluacion.index, self.evaluacion["inercia"], marker="o", color="#B81D24")
        axs[0].set_title("Método del codo (inercia)"); axs[0].set_xlabel("k"); axs[0].set_ylabel("Inercia")
        axs[1].plot(self.evaluacion.index, self.evaluacion["silhouette"], marker="o", color="#221F1F")
        axs[1].axvline(self.k, color="#B81D24", ls="--", label=f"k elegido = {self.k}")
        axs[1].set_title("Silhouette (mayor es mejor)"); axs[1].set_xlabel("k"); axs[1].legend()
        for a in axs:
            a.spines[["top", "right"]].set_visible(False)
        self._guardar(ruta, mostrar)

    def guardarGraficaGrupos(self, ruta, mostrar=False):
        pts = PCA(n_components=2, random_state=self.semilla).fit_transform(self.Xs)
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=self.titulos["grupo"], cmap="tab10", s=8, alpha=0.6)
        ax.legend(*sc.legend_elements(), title="Grupo", loc="best", fontsize=8)
        ax.set_title(f"Grupos de K-means (k = {self.k}) proyectados en 2 dimensiones (PCA)")
        ax.set_xlabel("Componente 1"); ax.set_ylabel("Componente 2")
        ax.spines[["top", "right"]].set_visible(False)
        self._guardar(ruta, mostrar)

    def guardarGraficaCentroides(self, ruta, mostrar=False):
        c = pd.DataFrame(self.modelo.cluster_centers_, columns=self.variables, index=range(1, self.k + 1))
        fig, ax = plt.subplots(figsize=(11, 0.6 * self.k + 2.5))
        im = ax.imshow(c.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(self.variables))); ax.set_xticklabels(self.variables, rotation=60, ha="right")
        ax.set_yticks(range(self.k)); ax.set_yticklabels([f"Grupo {i}" for i in c.index])
        for i in range(self.k):
            for j in range(len(self.variables)):
                ax.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(c.values[i, j]) > 1.5 else "black")
        ax.set_title("Perfil de cada grupo (centroides estandarizados: rojo = por encima del promedio)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        self._guardar(ruta, mostrar)
