import streamlit as st
import pandas as pd
from src.servicio import ServicioPrediccion
from src.comparadorModelos import ComparadorModelos
from src.modeloArbol import ModeloArbol
from src.modeloNaiveBayes import ModeloNaiveBayes
from src.modeloKNN import ModeloKNN

st.set_page_config(page_title="Netflix: catálogo y valoración", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Cargando datos y entrenando los modelos...")
def cargar():
    return ServicioPrediccion()


@st.cache_data(show_spinner="Comparando modelos (unos 20 segundos)...")
def comparar():
    s = cargar()
    c = ComparadorModelos(s.X, s.y)
    c.agregar("Árbol de decisión", ModeloArbol().estimador())
    c.agregar("Naive Bayes", ModeloNaiveBayes().estimador())
    c.agregar("KNN", ModeloKNN().estimador())
    return c.comparar().round(3), c.pruebaPareada("Árbol de decisión").round(3), c.lineaBase


s = cargar()
st.title("🎬 Catálogo de Netflix y valoración de usuarios")
st.caption("Proyecto de Minería de Datos · UNIMINUTO Ibagué · James Lozano y Elian Sandoval")

pestanas = st.tabs(["🔮 Predicción", "📊 Catálogo", "🧩 Grupos (K-means)", "⚖️ Modelos"])

with pestanas[0]:
    st.subheader("¿Tendrá un título una valoración alta (85 o más)?")
    st.write("Describe un título y los modelos estiman su valoración. **Naive Bayes** predice y el **árbol** explica el porqué.")
    c1, c2 = st.columns(2)
    with c1:
        tipo = st.radio("Tipo de contenido", ["Serie", "Película"], horizontal=True)
        if tipo == "Serie":
            temporadas = st.slider("Temporadas", 1, 17, 3)
            minutos = 0
        else:
            minutos = st.slider("Duración (minutos)", 40, 220, 100)
            temporadas = 0
        anio = st.slider("Año de lanzamiento", 1990, 2021, 2015)
    with c2:
        clasificacion = st.selectbox("Clasificación", list(s.clasificaciones().keys()), index=list(s.clasificaciones().keys()).index("TV-14") if "TV-14" in s.clasificaciones() else 0)
        generos = st.multiselect("Géneros", s.generos(), default=[])
    fila = s.construirFila(tipo, temporadas, minutos, anio, clasificacion, generos)
    r = s.predecir(fila)
    m1, m2 = st.columns(2)
    m1.metric("Naive Bayes: probabilidad de valoración alta", f"{r['bayes'] * 100:.0f} %")
    m2.metric("Árbol: probabilidad de valoración alta", f"{r['arbol'] * 100:.0f} %")
    st.markdown("**Cómo decide el árbol, paso a paso:**")
    for pregunta, respuesta in r["camino"]:
        st.write(f"- {pregunta} → **{respuesta}**")
    st.warning("Los modelos se entrenaron con solo 137 títulos y aciertan alrededor de 60 % (adivinar la clase más común da 54 %). "
               "Úsalos como una orientación, no como una predicción confiable.")

with pestanas[1]:
    st.subheader("Composición y crecimiento del catálogo")
    cat = s.catalogo.drop_duplicates("title")
    a, b = st.columns(2)
    a.write("**Películas y series**")
    a.bar_chart(cat["type"].value_counts())
    b.write("**Títulos agregados por año**")
    b.line_chart(cat.dropna(subset=["date_added"]).groupby(cat["date_added"].dt.year).size())
    generos = cat["listed_in"].str.split(", ").explode().str.strip().value_counts().head(10)
    st.write("**Géneros más frecuentes**")
    st.bar_chart(generos)

with pestanas[2]:
    st.subheader(f"El catálogo en {s.kmeans.k} grupos de títulos parecidos")
    st.dataframe(s.perfilesGrupos[["nombre", "titulos", "porcentaje", "pct_series", "anio_medio", "pct_infantil", "pct_adulto"]].round(1),
                 use_container_width=True)
    st.info("Estructura débil (silhouette 0,35): los grupos son segmentos descriptivos, no categorías con fronteras nítidas.")
    texto = st.text_input("Busca un título para ver a qué grupo pertenece", "Stranger")
    if texto:
        st.dataframe(s.buscarTitulo(texto), use_container_width=True)

with pestanas[3]:
    st.subheader("Árbol de decisión vs. Naive Bayes vs. KNN")
    if st.button("Calcular comparación"):
        tabla, pareada, base = comparar()
        st.dataframe(tabla, use_container_width=True)
        st.caption(f"Línea base (adivinar la clase más común): {base:.3f}. Validación cruzada repetida 5 × 10 con los mismos pliegues.")
        st.write("Diferencia frente al árbol (prueba pareada):")
        st.dataframe(pareada, use_container_width=True)
        st.success("Ninguna diferencia es estadísticamente significativa: usamos Naive Bayes para predecir y el árbol para explicar.")
