# Sesión 4 · Minería de Datos · UNIMINUTO Ibagué
# Transformación y reducción: escalado, codificación, discretización y PCA
# Adaptado al proyecto: Análisis Exploratorio del Catálogo de Netflix
# Ejecutar desde esta carpeta:  python3 transformaciones.py

from pathlib import Path
import warnings

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ajustar esta ruta según dónde tengas el CSV dentro de tu proyecto
# (por ejemplo: Path(__file__).resolve().parent / "CSV" / "resultados" / "netflix_dataset_procesado.csv")
RUTA = Path(__file__).resolve().parent / \
    "../CSV/resultados/netflix_dataset_procesado.csv"

# Columnas numéricas del dataset ya enriquecido por el pipeline
# (release_year y duration vienen del dataset original; durationValue,
# added_year y antiguedad_al_agregar son las que calculó AnalizadorDatos)
NUMERICAS = ["release_year", "durationValue",
             "added_year", "antiguedad_al_agregar"]

df = pd.read_csv(RUTA)

# added_year y antiguedad_al_agregar tienen nulos (títulos sin date_added
# válido); para escalar y aplicar PCA se necesita una matriz sin NaN
df = df.dropna(subset=NUMERICAS).reset_index(drop=True)

X = df[NUMERICAS]
print("Dataset limpio:", df.shape, "| matriz numérica:", X.shape)


# 1 · POR QUÉ ESCALAR: la distancia la domina la variable con unidades grandes
def distancia(a, b):
    return ((a - b) ** 2).sum() ** 0.5


c1, c2 = X.iloc[0], X.iloc[1]
print("\n--- Distancia entre dos títulos SIN escalar ---")
print(f"  ({df.loc[0, 'title']!r} vs {df.loc[1, 'title']!r})")
for col in NUMERICAS:
    print(f"  {col:<22} aporta {abs(c1[col] - c2[col]):>10.1f}")
print("Distancia total sin escalar :", round(distancia(c1, c2), 1))

X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=NUMERICAS)
c1e, c2e = X_minmax.iloc[0], X_minmax.iloc[1]
print("Distancia total con MinMax  :", round(distancia(c1e, c2e), 3))

# 2 · LOS TRES ESCALADORES
print("\n--- MinMaxScaler (todo a [0, 1]) ---")
print("Antes : min", X.min().round(1).tolist())
print("Después: min", X_minmax.min().round(3).tolist(),
      "max", X_minmax.max().round(3).tolist())

print("\n--- StandardScaler (media=0, desviación=1) ---")
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=NUMERICAS)
print("Medias tras escalar:", X_std.mean().round(3).tolist())
print("Desv. estándar     :", X_std.std().round(3).tolist())

print("\n--- RobustScaler (usa mediana e IQR, resiste outliers) ---")
X_rob = pd.DataFrame(RobustScaler().fit_transform(X), columns=NUMERICAS)
print("Mediana tras escalar:", X_rob.median().round(3).tolist())

# 3 · CODIFICACIÓN DE CATEGÓRICAS: one-hot
print("\n--- One-hot con get_dummies ---")
categoricas = ["type", "rating"]
X_cat = pd.get_dummies(df[categoricas], prefix=categoricas, dtype=int)
print("Columnas nuevas:", X_cat.columns.tolist())
print(X_cat.head(3).to_string())

# 4 · DISCRETIZACIÓN de release_year en 4 rangos por cuantiles
print("\n--- KBinsDiscretizer: release_year -> 4 categorías ---")
discretizador = KBinsDiscretizer(
    n_bins=4, encode="ordinal", strategy="quantile")
anio_bins = discretizador.fit_transform(df[["release_year"]]).ravel()
bordes = discretizador.bin_edges_[0].round(0)
print("Bordes de los rangos (años):", bordes)
print(pd.Series(anio_bins).value_counts().sort_index().to_string())

# 5 · PCA sobre las numéricas estandarizadas
print("\n--- PCA: ¿cuántas dimensiones resumen los datos? ---")
pca = PCA()
componentes = pca.fit_transform(X_std)
razones = pca.explained_variance_ratio_
acumulada = razones.cumsum().round(3)
print("Varianza explicada por componente :", razones.round(3)[:5])
print("Varianza acumulada                :", acumulada[:5])
para_80 = int((acumulada < 0.80).sum()) + 1
para_90 = int((acumulada < 0.90).sum()) + 1
print(
    f"Componentes para explicar >=80%: {para_80} | para >=90%: {para_90} (de {len(NUMERICAS)})")

# 6 · MATRIZ DE FEATURES LISTA PARA MODELAR
X_final = pd.concat([X_std, X_cat], axis=1)
print("\n--- Matriz final para modelado ---")
print("Forma:", X_final.shape)
print("Columnas:", X_final.columns.tolist())


# Distancia sin escalar: entre los dos primeros títulos, durationValue aporta 88 de los 88 puntos de distancia total — las demás columnas casi no cuentan porque están en escalas de años (miles) vs. minutos/temporadas. Es el mismo efecto que el ejemplo original de telecom, con tus datos.

#  One-hot: genera 17 columnas nuevas (2 de type + 15 de rating).

# Discretización: los 4 rangos de release_year quedan en 1925-2013, 2013-2017, 2017-2019 y 2019-2021 — coherente con lo que ya vimos del crecimiento acelerado del catálogo desde 2016.

# PCA — hallazgo interesante: el cuarto componente explica 0% de varianza. Esto no es un error: pasa porque antiguedad_al_agregar = added_year - release_year es una combinación lineal exacta de las otras dos columnas, así que en la práctica solo hay 3 dimensiones de información independientes, no 4. Con 3 componentes explicás el 100% de la varianza. Es un buen punto para mencionar si te preguntan sobre el PCA en la sustentación: demuestra que entendiste por qué salió ese resultado y no lo tomaste como un output ciego.

# Matriz final: 8.709 filas × 21 columnas (4 numéricas escaladas + 17 de one-hot).
