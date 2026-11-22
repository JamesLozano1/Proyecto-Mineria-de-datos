import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split


class ComparadorModelos:
    """
    Compara varios clasificadores de forma justa: mismos datos, misma partición
    70/30 y los mismos pliegues de validación cruzada repetida (5 x 10).
    """

    def __init__(self, X, y, semilla=42, pliegues=5, repeticiones=10):
        self.X, self.y = X, y
        self.semilla = semilla
        self.pliegues = pliegues
        self.cv = RepeatedStratifiedKFold(n_splits=pliegues, n_repeats=repeticiones, random_state=semilla)
        self.modelos = {}
        self.puntajes = {}
        self.tabla = None
        self.lineaBase = max(y.mean(), 1 - y.mean())

    def agregar(self, nombre, estimador):
        self.modelos[nombre] = estimador

    def comparar(self):
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.30, stratify=self.y, random_state=self.semilla)
        filas = []
        for nombre, estimador in self.modelos.items():
            r = cross_validate(estimador, self.X, self.y, cv=self.cv,
                               scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)
            self.puntajes[nombre] = r["test_accuracy"]
            prueba = clone(estimador).fit(Xtr, ytr).score(Xte, yte)
            filas.append({
                "modelo": nombre,
                "exactitud_cv": r["test_accuracy"].mean(),
                "desviacion": r["test_accuracy"].std(),
                "f1_alta": r["test_f1"].mean(),
                "auc": r["test_roc_auc"].mean(),
                "exactitud_entrenamiento": r["train_accuracy"].mean(),
                "sobreajuste": r["train_accuracy"].mean() - r["test_accuracy"].mean(),
                "exactitud_prueba_70_30": prueba,
            })
        self.tabla = pd.DataFrame(filas).set_index("modelo")
        return self.tabla

    def pruebaPareada(self, base):
        """
        Diferencia de cada modelo contra `base` en los mismos pliegues, con la
        corrección de Nadeau-Bengio (los pliegues repetidos no son independientes).
        """
        razon = 1 / (self.pliegues - 1)          # n_prueba / n_entrenamiento
        filas = []
        for nombre, p in self.puntajes.items():
            if nombre == base:
                continue
            d = p - self.puntajes[base]
            t = d.mean() / np.sqrt((1 / len(d) + razon) * d.var(ddof=1))
            filas.append({"modelo": nombre, "diferencia_media": d.mean(),
                          "p_valor": 2 * stats.t.sf(abs(t), len(d) - 1), "pliegues_que_gana": (d > 0).mean()})
        return pd.DataFrame(filas).set_index("modelo")

    def mejor(self):
        return self.tabla["exactitud_cv"].idxmax()

    def guardarGrafica(self, ruta, mostrar=False):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        nombres = list(self.tabla.index)
        colores = ["#B81D24", "#221F1F", "#8C8C8C", "#6B645E"][:len(nombres)]
        ax.bar(nombres, self.tabla["exactitud_cv"] * 100, yerr=self.tabla["desviacion"] * 100, capsize=6, color=colores, width=0.55)
        ax.axhline(self.lineaBase * 100, color="#B81D24", ls="--", lw=1.5, label="Adivinar la clase más común (%.1f %%)" % (self.lineaBase * 100))
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        for i, v in enumerate(self.tabla["exactitud_cv"] * 100):
            ax.text(i, v + 1.0, f"{v:.1f} %", ha="center", fontsize=11, fontweight="bold", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=1))
        ax.set_ylim(40, 80)
        ax.set_ylabel("Exactitud (validación cruzada, %)")
        ax.set_title("Comparación de modelos (barras de error: ± 1 desviación)")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        if mostrar:
            plt.show()
        else:
            plt.close()
