import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


class ModeloKNN:
    """
    K vecinos más cercanos: clasifica un título según la clase de los k títulos
    más parecidos. Como mide distancias, SÍ necesita escalar las variables
    (el año, ~2.000, dominaría sobre las variables 0/1).
    El escalador va dentro de un Pipeline para que se ajuste solo con los datos
    de entrenamiento de cada pliegue (sin filtrar información del conjunto de prueba).
    """

    VECINOS = [1, 3, 5, 7, 9, 11, 15, 21]

    def __init__(self, escalador="standard"):
        self.escalador = escalador

    def _escalador(self):
        if self.escalador == "standard":
            return StandardScaler()
        if self.escalador == "minmax":
            return MinMaxScaler()
        return "passthrough"

    def estimador(self, k=None):
        """Con k fijo devuelve el Pipeline; sin k elige el mejor k con validación interna."""
        tuberia = Pipeline([("escalador", self._escalador()), ("knn", KNeighborsClassifier())])
        if k is not None:
            return tuberia.set_params(knn__n_neighbors=k)
        return GridSearchCV(tuberia, {"knn__n_neighbors": self.VECINOS},
                            cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="accuracy")

    def curvaK(self, X, y, cv):
        """Exactitud en validación cruzada para cada k."""
        return pd.Series({k: cross_val_score(self.estimador(k), X, y, cv=cv).mean() for k in self.VECINOS})

    def efectoEscalamiento(self, X, y, cv, k):
        """Compara KNN sin escalar, con StandardScaler y con MinMaxScaler (mismo k y mismos pliegues)."""
        filas = {}
        for nombre, esc in [("Sin escalar", None), ("StandardScaler", "standard"), ("MinMaxScaler", "minmax")]:
            puntajes = cross_val_score(ModeloKNN(esc).estimador(k), X, y, cv=cv)
            filas[nombre] = {"exactitud": puntajes.mean(), "desviacion": puntajes.std()}
        return pd.DataFrame(filas).T
