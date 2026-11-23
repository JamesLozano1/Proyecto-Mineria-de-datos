import pandas as pd
import streamlit as st
from src import graficos as gr
from src.comparadorModelos import ComparadorModelos
from src.modeloArbol import ModeloArbol
from src.modeloKNN import ModeloKNN
from src.modeloNaiveBayes import ModeloNaiveBayes
from src.servicio import ServicioPrediccion

st.set_page_config(page_title="Netflix: catálogo y valoración", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
h1{letter-spacing:-0.5px}
.paso{display:flex;align-items:center;gap:14px;background:#1F1F1F;border:1px solid #2E2E2E;border-radius:12px;padding:12px 16px;margin-bottom:8px}
.paso .num{background:#E50914;color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;flex:none}
.paso .preg{flex:1;font-size:16px}
.paso .si{color:#4ADE80;font-weight:700}
.paso .no{color:#FF6B6B;font-weight:700}
.hoja{background:#1F1F1F;border-left:4px solid #E50914;border-radius:8px;padding:12px 16px;margin-top:6px}
</style>
""", unsafe_allow_html=True)


def mostrar(grafica):
    """Compatible con distintas versiones de Streamlit."""
    try:
        st.altair_chart(grafica, width="stretch")
    except (TypeError, ValueError):
        st.altair_chart(grafica, use_container_width=True)


def tabla(df, **opciones):
    try:
        st.dataframe(df, width="stretch", **opciones)
    except (TypeError, ValueError):
        st.dataframe(df, use_container_width=True, **opciones)


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
ind = s.indicadores()

with st.sidebar:
    st.markdown("## 🎬 Proyecto de minería de datos")
    st.write("Análisis del catálogo de Netflix y de la valoración de los usuarios.")
    st.markdown("**Integrantes**  \nJames Lozano  \nElian Sandoval")
    st.caption("UNIMINUTO · Ingeniería de Sistemas · Ibagué")
    st.divider()
    st.markdown("**Metodología:** CRISP-DM  \n**Modelos:** árbol de decisión, Naive Bayes, KNN y K-means")
    st.markdown("[Repositorio en GitHub](https://github.com/JamesLozano1/Proyecto-Mineria-de-datos)")
    st.divider()
    st.info("Los modelos se entrenaron con solo 137 títulos valorados. Son una orientación, no una predicción confiable.")

st.title("Catálogo de Netflix y valoración de usuarios")
st.caption("¿Qué características del contenido se asocian con una mejor valoración?")

k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.metric("Títulos en el catálogo", f"{ind['titulos']:,}".replace(",", "."))
with k2.container(border=True):
    st.metric("Películas / Series", f"{ind['peliculas'] * 100:.0f} % / {ind['series'] * 100:.0f} %")
with k3.container(border=True):
    st.metric("Títulos con valoración", ind["valorados"])
with k4.container(border=True):
    st.metric("Valoración promedio", f"{ind['valoracion']:.1f} / 100")

pestanas = st.tabs(["🔮 Predicción", "📊 Catálogo y valoración", "🧩 Grupos (K-means)", "⚖️ Modelos"])

# ------------------------------------------------------------------ Predicción
with pestanas[0]:
    st.subheader("¿Tendrá un título una valoración alta (85 o más)?")
    st.caption("Describe un título: Naive Bayes estima la probabilidad y el árbol de decisión explica el porqué, paso a paso.")
    izq, der = st.columns([1, 1.15], gap="large")
    clasif = s.clasificaciones()
    with izq, st.container(border=True):
        st.markdown("#### Describe el título")
        tipo = st.radio("Tipo de contenido", ["Serie", "Película"], horizontal=True)
        if tipo == "Serie":
            temporadas, minutos = st.slider("Temporadas", 1, 17, 3), 0
        else:
            minutos, temporadas = st.slider("Duración (minutos)", 40, 220, 100), 0
        anio = st.slider("Año de lanzamiento", 1990, 2021, 2015)
        lista = list(clasif.keys())
        clasificacion = st.selectbox("Clasificación", lista, index=lista.index("TV-14") if "TV-14" in lista else 0)
        generos = st.multiselect("Géneros", s.generos(), default=[])

    fila = s.construirFila(tipo, temporadas, minutos, anio, clasificacion, generos)
    r = s.predecir(fila)
    with der, st.container(border=True):
        st.markdown("#### Resultado")
        m1, m2 = st.columns(2)
        m1.metric("Naive Bayes (recomendado para predecir)", f"{r['bayes'] * 100:.0f} %")
        m2.metric("Árbol de decisión (explica)", f"{r['arbol'] * 100:.0f} %")
        st.progress(min(max(r["bayes"], 0.0), 1.0), text="Naive Bayes: probabilidad de valoración alta")
        st.progress(min(max(r["arbol"], 0.0), 1.0), text="Árbol: probabilidad de valoración alta")
        if r["bayes"] >= 0.6:
            st.success("Probabilidad **alta** de valoración de 85 o más.")
        elif r["bayes"] <= 0.4:
            st.warning("Probabilidad **baja** de valoración de 85 o más.")
        else:
            st.info("Resultado **incierto**: los modelos no se inclinan con claridad.")
        st.caption("Con 137 títulos los modelos aciertan cerca de 60 % (adivinar la clase más común da 54 %).")

    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        st.markdown("#### Cómo decide el árbol, paso a paso")
        for i, (pregunta, respuesta) in enumerate(r["camino"], 1):
            clase = "si" if respuesta == "Sí" else "no"
            st.markdown(f'<div class="paso"><span class="num">{i}</span><span class="preg">{pregunta}</span>'
                        f'<span class="{clase}">{respuesta}</span></div>', unsafe_allow_html=True)
        n, altas = s.arbol.resumenHoja(fila)
        st.markdown(f'<div class="hoja">Resultado del árbol: de <b>{n}</b> títulos parecidos, <b>{altas}</b> '
                    f'({altas / n * 100:.0f} %) tuvieron valoración alta.</div>', unsafe_allow_html=True)
        if n <= 3:
            st.warning("Este camino se basa en muy pocos títulos: poco confiable.")
    with c2:
        sens, etiqueta = s.sensibilidad(tipo, anio, clasificacion, generos, temporadas, minutos)
        mostrar(gr.sensibilidad(sens, temporadas if tipo == "Serie" else minutos, etiqueta))
        st.caption("Los dos modelos pueden dar probabilidades muy distintas: Naive Bayes cambia de forma gradual y el árbol por saltos.")
    mostrar(gr.barras_h(s.importanciaArbol(), "Variables que más usa el árbol", formato=".0%", alto=280))

# ------------------------------------------------------------------ Catálogo y valoración
with pestanas[1]:
    cat = s.catalogo.drop_duplicates("title")
    st.subheader("El catálogo")
    a, b = st.columns([1, 2], gap="large")
    with a:
        mostrar(gr.donut_tipo(s.catalogo))
    with b:
        mostrar(gr.area_anual(s.catalogo))
    paises = cat.assign(c=cat["country"].str.split(", ")).explode("c")
    paises = paises.assign(c=paises["c"].str.strip())
    paises = paises[paises["c"] != "Desconocido"]["c"].value_counts().head(10)
    generos_top = cat["listed_in"].str.split(", ").explode().str.strip().value_counts().head(10)
    c, d = st.columns(2, gap="large")
    with c:
        mostrar(gr.barras_h(paises, "Principales países productores"))
    with d:
        mostrar(gr.barras_h(generos_top, "Géneros más frecuentes", color=gr.CLARO))
    e, f = st.columns(2, gap="large")
    with e:
        mostrar(gr.hist_duracion(s.catalogo))
    with f:
        mostrar(gr.barras_h(cat["rating"].value_counts().head(8), "Clasificación de audiencia", alto=260))

    st.subheader("La valoración de los usuarios")
    st.caption(f"Basada en {ind['valorados']} títulos con puntuación (de 500 valoraciones únicas).")
    g1, g2 = st.columns([1.5, 1], gap="large")
    with g1:
        mostrar(gr.valoracion_genero(s.valoracionPorGenero(), ind["valoracion"]))
    with g2:
        mostrar(gr.box_audiencia(s.valoracionAudiencia()))
        st.caption("El contenido infantil se valora menos (74,5) que el adolescente/adulto (82,7).")

# ------------------------------------------------------------------ Grupos
with pestanas[2]:
    st.subheader(f"El catálogo en {s.kmeans.k} grupos de títulos parecidos")
    st.caption("K-means agrupa los títulos por tipo, duración, año, clasificación y género. La estructura es débil (silhouette 0,35): "
               "son segmentos descriptivos, no categorías con fronteras nítidas.")
    mostrar(gr.dispersion_grupos(s.coordenadasGrupos()))
    st.caption("Haz clic en la leyenda para resaltar un grupo; pasa el cursor sobre un punto para ver el título.")
    perfiles = s.perfilesGrupos[["nombre", "titulos", "porcentaje", "pct_series", "pct_infantil", "pct_adulto", "anio_medio"]].round(1)
    perfiles.index = ["Grupo %d" % i for i in perfiles.index]
    tabla(perfiles, column_config={
        "nombre": "Perfil", "titulos": "Títulos",
        "porcentaje": st.column_config.ProgressColumn("% del catálogo", min_value=0, max_value=30, format="%.1f"),
        "pct_series": st.column_config.ProgressColumn("% series", min_value=0, max_value=100, format="%.0f"),
        "pct_infantil": st.column_config.ProgressColumn("% infantil", min_value=0, max_value=100, format="%.0f"),
        "pct_adulto": st.column_config.ProgressColumn("% adulto", min_value=0, max_value=100, format="%.0f"),
        "anio_medio": st.column_config.NumberColumn("Año medio", format="%.0f")})
    mostrar(gr.heatmap_centroides(s.centroides()))
    texto = st.text_input("Busca un título para ver a qué grupo pertenece", "Stranger")
    if texto:
        resultado = s.buscarTitulo(texto)
        tabla(resultado.rename(columns={"title": "Título", "type": "Tipo", "release_year": "Año", "listed_in": "Géneros", "grupo": "Grupo"}))

# ------------------------------------------------------------------ Modelos
with pestanas[3]:
    st.subheader("Árbol de decisión vs. Naive Bayes vs. KNN")
    st.caption("Mismos 137 títulos, misma partición y los mismos pliegues de validación cruzada repetida (5 × 10).")
    if st.button("Calcular comparación", type="primary"):
        st.session_state["comparacion"] = comparar()
    if "comparacion" in st.session_state:
        resumen, pareada, base = st.session_state["comparacion"]
        m1, m2, m3 = st.columns(3, gap="large")
        with m1:
            mostrar(gr.comparacion(resumen, "exactitud_cv", "Exactitud (validación cruzada)", base=base, escala=[0, 0.8]))
            st.caption(f"Línea azul: adivinar la clase más común ({base * 100:.1f} %).")
        with m2:
            mostrar(gr.comparacion(resumen, "auc", "AUC (qué tan bien ordena)", formato=".2f", escala=[0, 1]))
        with m3:
            mostrar(gr.comparacion(resumen, "sobreajuste", "Sobreajuste (menos es mejor)"))
        with st.expander("Ver las tablas completas"):
            tabla(resumen)
            st.write("Diferencia frente al árbol (prueba pareada):")
            tabla(pareada)
        st.success("Ninguna diferencia es estadísticamente significativa: usamos Naive Bayes para predecir y el árbol para explicar.")
    else:
        st.info("Pulsa el botón para comparar los tres modelos (tarda unos 20 segundos).")
