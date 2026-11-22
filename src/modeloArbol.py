import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold


class ModeloArbol:
    """
    Árbol de decisión que predice si un título tiene valoración alta
    (user_rating_score >= umbral) a partir de su tipo, duración, año,
    clasificación y género. Usa los títulos de netflix.csv que cruzan
    por nombre con netflix_titles.csv.
    """

    INFANTIL = ["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"]

    def __init__(self, maxProfundidad=4, umbral=85, semilla=42, minTitulosGenero=5):
        self.maxProfundidad = maxProfundidad
        self.umbral = umbral
        self.semilla = semilla
        self.minTitulosGenero = minTitulosGenero
        self.modelo = None

    def prepararDatos(self, valoraciones, catalogo):
        valoraciones = valoraciones.dropna(subset=["user_rating_score"])
        catalogoUnico = catalogo.drop_duplicates("title")[["title", "type", "listed_in", "duration"]]
        m = valoraciones.merge(catalogoUnico, on="title", how="inner").copy()

        m["alta"] = (m["user_rating_score"] >= self.umbral).astype(int)
        m["es_serie"] = (m["type"] == "TV Show").astype(int)
        duracion = pd.to_numeric(m["duration"].str.split(" ").str[0], errors="coerce")
        m["temporadas"] = np.where(m["es_serie"] == 1, duracion, 0)
        m["minutos"] = np.where(m["es_serie"] == 0, duracion, 0)
        m["infantil"] = m["rating"].isin(self.INFANTIL).astype(int)

        generos = m.assign(g=m["listed_in"].str.split(", ")).explode("g")
        generos["g"] = generos["g"].str.strip()
        conteo = generos.groupby("g")["title"].nunique()
        frecuentes = conteo[conteo >= self.minTitulosGenero].index.tolist()
        for g in frecuentes:
            m["g_" + g] = m["listed_in"].str.contains(g, regex=False).astype(int)

        self.variables = ["release_year", "rating_level", "es_serie", "temporadas", "minutos", "infantil"] \
            + ["g_" + g for g in frecuentes]
        self.X = m[self.variables].fillna(0)
        self.y = m["alta"]
        return self.X, self.y

    def estimador(self):
        """Árbol sin entrenar, con la misma configuración (para compararlo con otros modelos)."""
        return DecisionTreeClassifier(max_depth=self.maxProfundidad, random_state=self.semilla)

    def entrenarCompleto(self):
        """Entrena con todos los títulos (es el modelo que se despliega; la evaluación usa entrenar())."""
        self.modelo = self.estimador().fit(self.X, self.y)
        return self.modelo

    @staticmethod
    def pregunta(variable, umbral):
        """Convierte un corte técnico del árbol en una pregunta de sí/no."""
        t = int(np.floor(umbral))
        if variable == "temporadas":
            return "¿Tiene más de %d %s?" % (t, "temporada" if t == 1 else "temporadas")
        if variable == "minutos":
            return "¿Es una película de más de %d minutos?" % t
        if variable == "release_year":
            return "¿Se estrenó después de %d?" % t
        if variable == "infantil":
            return "¿Es contenido infantil o familiar?"
        if variable == "rating_level":
            return "¿Su nivel de clasificación (rating_level) supera %d?" % t
        nombres = {"Crime TV Shows": "una serie de crimen", "Romantic TV Shows": "una serie romántica"}
        g = variable[2:]
        return "¿Es %s?" % nombres.get(g, "del género " + g)

    def explicarCamino(self, fila):
        """Lista de (pregunta, respuesta) que recorre el árbol para una fila de variables."""
        arbol = self.modelo.tree_
        nodo, camino = 0, []
        while arbol.children_left[nodo] != -1:
            variable = self.variables[arbol.feature[nodo]]
            va_derecha = fila[variable] > arbol.threshold[nodo]
            camino.append((self.pregunta(variable, arbol.threshold[nodo]), "Sí" if va_derecha else "No"))
            nodo = arbol.children_right[nodo] if va_derecha else arbol.children_left[nodo]
        return camino

    def entrenar(self):
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            self.X, self.y, test_size=0.30, stratify=self.y, random_state=self.semilla)
        self.modelo = DecisionTreeClassifier(max_depth=self.maxProfundidad, random_state=self.semilla)
        self.modelo.fit(self.Xtr, self.ytr)
        return self.modelo

    def evaluar(self):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=self.semilla)
        puntajes = cross_val_score(
            DecisionTreeClassifier(max_depth=self.maxProfundidad, random_state=self.semilla), self.X, self.y, cv=cv)
        return {
            "titulos": len(self.X),
            "variables": len(self.variables),
            "profundidad": self.modelo.get_depth(),
            "hojas": int(self.modelo.get_n_leaves()),
            "exactitud_entrenamiento": self.modelo.score(self.Xtr, self.ytr),
            "exactitud_prueba": self.modelo.score(self.Xte, self.yte),
            "linea_base_prueba": max(self.yte.mean(), 1 - self.yte.mean()),
            "validacion_cruzada": puntajes.mean(),
            "validacion_cruzada_desv": puntajes.std(),
            "linea_base": max(self.y.mean(), 1 - self.y.mean()),
        }

    def reglas(self):
        return export_text(self.modelo, feature_names=self.variables)

    def importancias(self):
        imp = pd.Series(self.modelo.feature_importances_, index=self.variables)
        return imp[imp > 0].sort_values(ascending=False)

    def guardarGrafica(self, ruta, mostrar=False):
        fig, ax = plt.subplots(figsize=(26, 11))
        plot_tree(self.modelo, feature_names=self.variables, class_names=["Baja (<%d)" % self.umbral, "Alta (>=%d)" % self.umbral],
                  filled=True, rounded=True, impurity=False, fontsize=9, ax=ax)
        ax.set_title("Árbol de decisión (profundidad máxima %d)" % self.maxProfundidad, fontsize=15)
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        if mostrar:
            plt.show()
        else:
            plt.close()
