import html

import altair as alt
import pandas as pd
import streamlit as st

from src.servicio import ServicioPrediccion
from src.comparadorModelos import ComparadorModelos
from src.modeloArbol import ModeloArbol
from src.modeloNaiveBayes import ModeloNaiveBayes
from src.modeloKNN import ModeloKNN

st.set_page_config(page_title="Netflix: catálogo y valoración",
                   page_icon="🎬", layout="wide")

# Streamlit nuevo usa width="stretch"; las versiones anteriores usan use_container_width
_version = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
ANCHO = {"width": "stretch"} if _version >= (
    1, 50) else {"use_container_width": True}

# ---------------------------------------------------------------- paleta
ROJO = "#E50914"
AMBAR = "#F2B84B"
CREMA = "#F3EEE8"
APAGADO = "#A89F9A"
LINEA = "#3A3236"

# ---------------------------------------------------------------- estilos
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="st-"], .stMarkdown { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* Cabecera */
.hero {
    position: relative; overflow: hidden;
    padding: 2rem 2.2rem 1.8rem 2.2rem; margin-bottom: 1.4rem;
    border-radius: 14px; background: #221C1E; border: 1px solid #3A3236;
}
.hero::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 8px;
    background: repeating-linear-gradient(to bottom, #E50914 0 22px, transparent 22px 34px);
}
.hero h1 {
    font-family: 'Bebas Neue', sans-serif; font-weight: 400; font-size: 3.4rem;
    letter-spacing: 0.04em; line-height: 1; margin: 0 0 0.5rem 0.6rem; color: #F3EEE8;
}
.hero p { margin: 0 0 0 0.6rem; color: #A89F9A; font-size: 1rem; line-height: 1.5; max-width: 62ch; }
.hero .autores { margin-top: 0.9rem; font-size: 0.85rem; color: #A89F9A; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid #3A3236; }
.stTabs [data-baseweb="tab"] { height: 3rem; padding: 0 1.1rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #F3EEE8; }

/* Medidores de probabilidad */
.med { background: #221C1E; border: 1px solid #3A3236; border-radius: 12px; padding: 1.1rem 1.3rem 1.2rem 1.3rem; height: 100%; }
.med-t { color: #A89F9A; font-size: 0.95rem; margin-bottom: 0.2rem; }
.med-v { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; line-height: 1; color: #F3EEE8; }
.med-v span { font-size: 1.8rem; color: #A89F9A; margin-left: 0.15rem; }
.med-bar { position: relative; height: 10px; border-radius: 5px; background: #3A3236; margin: 0.7rem 0 0.5rem 0; }
.med-bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; background: #E50914; }
.med-bar b { position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px; background: #F3EEE8; opacity: 0.55; }
.med-n { color: #A89F9A; font-size: 0.85rem; }
.med.principal { border-color: #E50914; }

/* Veredicto */
.veredicto { margin: 1rem 0 0.2rem 0; padding: 0.9rem 1.2rem; border-radius: 10px; background: #221C1E; border-left: 4px solid #F2B84B; font-size: 1.02rem; }
.veredicto.alto { border-left-color: #E50914; }
.veredicto small { display: block; color: #A89F9A; margin-top: 0.15rem; }

/* Camino del árbol */
ol.camino { list-style: none; counter-reset: paso; margin: 0.4rem 0 0 0; padding: 0; }
ol.camino li {
    counter-increment: paso; position: relative; padding: 0.55rem 0 0.55rem 2.6rem;
    border-left: 2px solid #3A3236; margin-left: 0.9rem;
}
ol.camino li::before {
    content: counter(paso); position: absolute; left: -0.95rem; top: 0.5rem;
    width: 1.8rem; height: 1.8rem; border-radius: 50%; background: #161213; border: 2px solid #E50914;
    display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700;
}
ol.camino li:last-child { border-left-color: transparent; }
ol.camino .resp { display: inline-block; margin-left: 0.4rem; padding: 0.05rem 0.6rem; border-radius: 999px; background: #E50914; color: #fff; font-weight: 700; font-size: 0.88rem; }

/* Indicadores */
.kpi { background: #221C1E; border: 1px solid #3A3236; border-radius: 12px; padding: 0.9rem 1.2rem; }
.kpi .n { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; line-height: 1.05; color: #F3EEE8; }
.kpi .t { color: #A89F9A; font-size: 0.9rem; }

/* Aviso */
.aviso { margin-top: 1.4rem; padding: 0.8rem 1.1rem; border-radius: 10px; background: rgba(242,184,75,0.10); border: 1px solid rgba(242,184,75,0.35); color: #F3EEE8; font-size: 0.92rem; }

.pie { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #3A3236; color: #A89F9A; font-size: 0.82rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- utilidades
def estilo(grafico, alto=320):
    """Aplica la identidad visual a un gráfico de Altair."""
    return (
        grafico.properties(height=alto)
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=APAGADO, titleColor=APAGADO, gridColor=LINEA, domainColor=LINEA, tickColor=LINEA,
                        labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor=CREMA, titleColor=APAGADO, orient="bottom")
    )


def medidor(titulo, p, nota, principal=False):
    pct = int(round(p * 100))
    clase = "med principal" if principal else "med"
    return (
        f'<div class="{clase}"><div class="med-t">{html.escape(titulo)}</div>'
        f'<div class="med-v">{pct}<span>%</span></div>'
        f'<div class="med-bar"><i style="width:{pct}%"></i><b></b></div>'
        f'<div class="med-n">{html.escape(nota)}</div></div>'
    )


def kpi(numero, texto):
    return f'<div class="kpi"><div class="n">{html.escape(str(numero))}</div><div class="t">{html.escape(texto)}</div></div>'


# ------------------------------------------------ gráficas (mismo diseño que VisualizadorDatos)
PALETA = [ROJO, AMBAR, "#C9C1BA", "#8A1118", "#7A6F6A"]
SIN_DATO = {"", "nan", "unknown", "desconocido",
            "no disponible", "n/a", "none"}


def _largo(datos):
    """Convierte una Serie o un DataFrame de pandas al formato largo que usa Altair."""
    if isinstance(datos, pd.Series):
        df = datos.rename("valor").rename_axis("categoria").reset_index()
        df["serie"] = str(datos.name) if datos.name is not None else "valor"
        return df, False
    df = datos.rename_axis("categoria").reset_index().melt(
        "categoria", var_name="serie", value_name="valor")
    return df, datos.shape[1] > 1


def _formatoNumero(df):
    """Eje con punto de miles si son enteros, un decimal si no lo son."""
    if (df["valor"].dropna() % 1 == 0).all():
        return {"labelExpr": "replace(format(datum.value, ',.0f'), /,/g, '.')"}, ",.0f"
    return {"format": ".1f"}, ",.2f"


def _titulo(texto):
    return alt.TitleParams(texto, anchor="start", color=CREMA, fontSize=15, fontWeight="bold", offset=14)


def graficaBarras(datos, titulo, etiquetaX, etiquetaY, alto=340):
    """Barras con título, ejes rotulados, categorías inclinadas 45° y el valor sobre cada barra."""
    df, multi = _largo(datos)
    df["categoria"] = df["categoria"].astype(str)
    ejeY, fmt = _formatoNumero(df)

    base = alt.Chart(df).encode(
        x=alt.X("categoria:N", sort=None, title=etiquetaX,
                axis=alt.Axis(labelAngle=-45, labelLimit=170)),
        y=alt.Y("valor:Q", title=etiquetaY, axis=alt.Axis(**ejeY)),
    )
    tooltip = [alt.Tooltip("categoria:N", title=etiquetaX), alt.Tooltip(
        "valor:Q", title=etiquetaY, format=fmt)]

    if multi:
        series = list(df["serie"].unique())
        barras = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            color=alt.Color("serie:N", title=None, scale=alt.Scale(
                domain=series, range=PALETA[: len(series)])),
            tooltip=tooltip + [alt.Tooltip("serie:N", title="Serie")],
        )
        if hasattr(alt, "XOffset"):
            barras = barras.encode(xOffset="serie:N")
        grafico = barras
    else:
        barras = base.mark_bar(color=ROJO, cornerRadiusTopLeft=3,
                               cornerRadiusTopRight=3).encode(tooltip=tooltip)
        grafico = barras
        if len(df) <= 15:
            if fmt == ",.0f":
                etiquetas = base.transform_calculate(etiqueta="replace(format(datum.valor, ',.0f'), /,/g, '.')").mark_text(
                    dy=-8, color=CREMA, fontSize=11).encode(text="etiqueta:N")
            else:
                etiquetas = base.mark_text(
                    dy=-8, color=CREMA, fontSize=11).encode(text=alt.Text("valor:Q", format=fmt))
            grafico = barras + etiquetas
    return grafico.properties(title=_titulo(titulo)), alto


def graficaLinea(datos, titulo, etiquetaX, etiquetaY, alto=340):
    """Línea con marcadores, cuadrícula, título y ejes rotulados."""
    df, multi = _largo(datos)
    tipo = "T" if pd.api.types.is_datetime64_any_dtype(
        df["categoria"]) else "O"
    ejeY, fmt = _formatoNumero(df)

    base = alt.Chart(df).encode(
        x=alt.X(f"categoria:{tipo}", sort=None, title=etiquetaX,
                axis=alt.Axis(labelAngle=0, grid=True)),
        y=alt.Y("valor:Q", title=etiquetaY, axis=alt.Axis(**ejeY)),
        tooltip=[alt.Tooltip(f"categoria:{tipo}", title=etiquetaX), alt.Tooltip(
            "valor:Q", title=etiquetaY, format=fmt)],
    )
    if multi:
        series = list(df["serie"].unique())
        linea = base.mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(filled=True, size=70)).encode(
            color=alt.Color("serie:N", title=None, scale=alt.Scale(
                domain=series, range=PALETA[: len(series)]))
        )
    else:
        linea = base.mark_line(color=ROJO, strokeWidth=2.6, point=alt.OverlayMarkDef(
            color=ROJO, filled=True, size=70))
    return linea.properties(title=_titulo(titulo)), alto


def mostrar(grafica, contenedor=st):
    """Dibuja un gráfico creado con graficaBarras o graficaLinea, con el tema de la aplicación."""
    grafico, alto = grafica
    contenedor.altair_chart(estilo(grafico, alto), **ANCHO, theme=None)


# ---------------------------------------------------------------- datos y modelos
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

st.markdown(
    """
<div class="hero">
  <h1>Catálogo de Netflix y valoración de usuarios</h1>
  <p>Explora cómo está compuesto el catálogo, descubre grupos de títulos parecidos y estima si un título nuevo
  tendría una valoración alta.</p>
  <div class="autores">Proyecto de Minería de Datos, UNIMINUTO Ibagué. James Lozano y Elian Sandoval.</div>
</div>
""",
    unsafe_allow_html=True,
)

pestanas = st.tabs(["Predicción", "Catálogo", "Grupos", "Modelos"])

# ================================================================ PREDICCIÓN
with pestanas[0]:
    st.subheader("¿Tendrá un título una valoración de 85 o más?")
    st.write("Describe un título y los modelos estiman su valoración. **Naive Bayes** predice y el **árbol** explica el porqué.")

    clasifs = list(s.clasificaciones().keys())
    idx_defecto = clasifs.index("TV-14") if "TV-14" in clasifs else 0

    with st.container(border=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            tipo = st.radio("Tipo de contenido", [
                            "Serie", "Película"], horizontal=True)
            if tipo == "Serie":
                temporadas = st.slider("Temporadas", 1, 17, 3)
                minutos = 0
            else:
                minutos = st.slider("Duración (minutos)", 40, 220, 100)
                temporadas = 0
            anio = st.slider("Año de lanzamiento", 1990, 2021, 2015)
        with c2:
            clasificacion = st.selectbox(
                "Clasificación", clasifs, index=idx_defecto)
            generos = st.multiselect("Géneros", s.generos(
            ), default=[], placeholder="Elige uno o varios géneros")

    fila = s.construirFila(tipo, temporadas, minutos,
                           anio, clasificacion, generos)
    r = s.predecir(fila)

    st.write("")
    m1, m2 = st.columns(2, gap="large")
    m1.markdown(medidor("Naive Bayes: probabilidad de valoración alta", r["bayes"], "Es el modelo que usamos para predecir.", principal=True),
                unsafe_allow_html=True)
    m2.markdown(medidor("Árbol: probabilidad de valoración alta", r["arbol"], "Es el modelo que usamos para explicar."),
                unsafe_allow_html=True)

    pb = r["bayes"]
    if pb >= 0.6:
        veredicto, clase, detalle = "Este perfil se parece a títulos con valoración alta.", "alto", "La señal es moderada: el modelo se equivoca con frecuencia."
    elif pb <= 0.4:
        veredicto, clase, detalle = "Este perfil se parece más a títulos con valoración por debajo de 85.", "", "La señal es moderada: el modelo se equivoca con frecuencia."
    else:
        veredicto, clase, detalle = "El modelo no ve una tendencia clara para este perfil.", "", "Prueba cambiar el género o la clasificación para ver cómo se mueve."
    st.markdown(
        f'<div class="veredicto {clase}">{veredicto}<small>{detalle}</small></div>', unsafe_allow_html=True)

    st.markdown("#### Cómo decide el árbol")
    pasos = "".join(
        f'<li>{html.escape(str(pregunta))}<span class="resp">{html.escape(str(respuesta))}</span></li>'
        for pregunta, respuesta in r["camino"]
    )
    st.markdown(f'<ol class="camino">{pasos}</ol>', unsafe_allow_html=True)

    st.markdown(
        '<div class="aviso">Los modelos se entrenaron con solo 137 títulos y aciertan alrededor de 60 % '
        '(adivinar la clase más común da 54 %). Úsalos como una orientación, no como una predicción confiable.</div>',
        unsafe_allow_html=True,
    )

# ================================================================ CATÁLOGO
with pestanas[1]:
    st.subheader("Composición y crecimiento del catálogo")
    cat = s.catalogo.drop_duplicates("title")

    es_serie = cat["type"].astype(str).str.contains(
        "TV|Serie", case=False, regex=True)
    con_fecha = cat.dropna(subset=["date_added"]).copy()
    con_fecha["anio"] = con_fecha["date_added"].dt.year.astype(int)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi(f"{len(cat):,}".replace(",", "."),
                "títulos en el catálogo"), unsafe_allow_html=True)
    k2.markdown(kpi(f"{es_serie.mean() * 100:.0f} %",
                "son series"), unsafe_allow_html=True)
    k3.markdown(kpi(f"{(~es_serie).mean() * 100:.0f} %",
                "son películas"), unsafe_allow_html=True)
    if "release_year" in cat.columns:
        k4.markdown(kpi(int(cat["release_year"].median()),
                    "año de lanzamiento típico"), unsafe_allow_html=True)
    else:
        k4.markdown(kpi(int(con_fecha["anio"].max(
        )), "último año con títulos agregados"), unsafe_allow_html=True)

    st.write("")

    # Datos de cada gráfica
    por_tipo = cat["type"].replace(
        {"Movie": "Película", "TV Show": "Serie"}).value_counts()
    por_anio = con_fecha.groupby("anio").size()
    generos_top = cat["listed_in"].str.split(
        ",").explode().str.strip().value_counts().head(10)

    # Fila 1: tipo y crecimiento
    f1a, f1b = st.columns([1, 2], gap="large")
    with f1a.container(border=True):
        mostrar(graficaBarras(por_tipo, "Películas y series",
                "Tipo de contenido", "Títulos", 320))
    with f1b.container(border=True):
        mostrar(graficaLinea(por_anio, "Títulos agregados por año",
                "Año en que se agregó", "Títulos", 320))

    # Fila 2: año de lanzamiento vs. año agregado, y antigüedad al agregar
    if "release_year" in cat.columns and not por_anio.empty:
        f2a, f2b = st.columns(2, gap="large")
        desde, hasta = int(por_anio.index.min()), int(por_anio.index.max())
        lanzados = cat["release_year"].dropna().astype(int).value_counts()
        comparacion = pd.DataFrame({"Año en que se agregó": por_anio,
                                   "Año de lanzamiento": lanzados}).loc[desde:hasta].fillna(0).astype(int)
        with f2a.container(border=True):
            mostrar(graficaLinea(
                comparacion, "Agregados a Netflix vs. lanzados originalmente", "Año", "Títulos", 340))

        antiguedad = (con_fecha["anio"] - con_fecha["release_year"]
                      ).clip(lower=0).groupby(con_fecha["anio"]).mean().round(1)
        antiguedad = antiguedad[con_fecha.groupby("anio").size() >= 10]
        with f2b.container(border=True):
            mostrar(graficaLinea(antiguedad, "Años entre el lanzamiento y la llegada a Netflix",
                    "Año en que se agregó", "Años (promedio)", 340))
            st.caption("Solo años con 10 títulos o más.")

    # Fila 3: géneros y países
    f3a, f3b = st.columns(2, gap="large")
    with f3a.container(border=True):
        mostrar(graficaBarras(generos_top,
                "Géneros más frecuentes", "Género", "Títulos", 400))
    if "country" in cat.columns:
        paises = cat["country"].dropna().astype(
            str).str.split(",").explode().str.strip()
        paises = paises[~paises.str.lower().isin(
            SIN_DATO)].value_counts().head(10)
        with f3b.container(border=True):
            mostrar(graficaBarras(
                paises, "Países con más títulos", "País", "Títulos", 400))

    # Fila 4: clasificación por edades
    if "rating" in cat.columns:
        clasif = cat["rating"].dropna().astype(str)
        clasif = clasif[~clasif.str.lower().isin(
            SIN_DATO)].value_counts().head(10)
        with st.container(border=True):
            mostrar(graficaBarras(clasif, "Clasificaciones más comunes",
                    "Clasificación", "Títulos", 340))

# ================================================================ GRUPOS
with pestanas[2]:
    st.subheader(f"El catálogo en {s.kmeans.k} grupos de títulos parecidos")
    perfiles = s.perfilesGrupos[["nombre", "titulos", "porcentaje",
                                 "pct_series", "anio_medio", "pct_infantil", "pct_adulto"]].copy()

    # Si los porcentajes vienen entre 0 y 1, se pasan a escala 0-100 para mostrarlos.
    cols_pct = ["porcentaje", "pct_series", "pct_infantil", "pct_adulto"]
    if perfiles[cols_pct].max().max() <= 1.0:
        perfiles[cols_pct] = perfiles[cols_pct] * 100
    perfiles = perfiles.round(1)

    g1, g2 = st.columns([3, 2], gap="large")
    with g1:
        def barra(titulo):
            return st.column_config.ProgressColumn(titulo, format="%.1f%%", min_value=0, max_value=100)

        st.dataframe(
            perfiles,
            **ANCHO,
            hide_index=True,
            column_config={
                "nombre": st.column_config.TextColumn("Grupo", width="medium"),
                "titulos": st.column_config.NumberColumn("Títulos", format="%d"),
                "porcentaje": barra("Del catálogo"),
                "pct_series": barra("Series"),
                "anio_medio": st.column_config.NumberColumn("Año medio", format="%d"),
                "pct_infantil": barra("Infantil"),
                "pct_adulto": barra("Adulto"),
            },
        )
    with g2:
        mostrar(graficaBarras(perfiles.set_index("nombre")[
                "titulos"], "Títulos por grupo", "Grupo", "Títulos", 420))

    st.info("Estructura débil (silhouette 0,35): los grupos son segmentos descriptivos, no categorías con fronteras nítidas.")

    st.markdown("#### ¿A qué grupo pertenece un título?")
    texto = st.text_input("Busca un título", "Stranger",
                          placeholder="Escribe parte del nombre")
    if texto:
        st.dataframe(s.buscarTitulo(texto), **ANCHO, hide_index=True)

# ================================================================ MODELOS
with pestanas[3]:
    st.subheader("Árbol de decisión vs. Naive Bayes vs. KNN")
    st.write("Los tres modelos se evalúan con los mismos datos y los mismos pliegues de validación cruzada, para que la comparación sea justa.")

    if st.button("Calcular comparación", type="primary"):
        st.session_state["comparado"] = True

    if st.session_state.get("comparado"):
        tabla, pareada, base = comparar()
        numericas = tabla.select_dtypes("number")

        t1, t2 = st.columns([3, 2], gap="large")
        with t1:
            st.markdown("**Resultados por modelo**")
            st.dataframe(tabla, **ANCHO)
            st.caption(
                f"Línea base (adivinar la clase más común): {base:.3f}. Validación cruzada repetida 5 × 10 con los mismos pliegues.")
        with t2:
            if not numericas.empty:
                metrica = st.selectbox(
                    "Métrica a graficar", list(numericas.columns))
                datos = tabla.reset_index()
                no_num = tabla.select_dtypes(exclude="number").columns
                col_modelo = no_num[0] if len(no_num) else datos.columns[0]
                barras = alt.Chart(datos).mark_bar(color=ROJO, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X(f"{col_modelo}:N", title=None,
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f"{metrica}:Q", title=metrica,
                            scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip(f"{col_modelo}:N", title="Modelo"), alt.Tooltip(
                        f"{metrica}:Q", format=".3f")],
                )
                capas = barras
                if any(p in str(metrica).lower() for p in ("exact", "acc")):
                    linea = alt.Chart(pd.DataFrame({"base": [base]})).mark_rule(
                        color=AMBAR, strokeDash=[6, 4], strokeWidth=2).encode(y="base:Q")
                    capas = barras + linea
                    st.caption("La línea ámbar marca la línea base.")
                st.altair_chart(estilo(capas, 280), **ANCHO, theme=None)

        st.markdown("**Diferencia frente al árbol (prueba pareada)**")
        st.dataframe(pareada, **ANCHO)
        st.success(
            "Ninguna diferencia es estadísticamente significativa: usamos Naive Bayes para predecir y el árbol para explicar.")
    else:
        st.info("Pulsa el botón para calcular la comparación. Tarda unos 20 segundos.")

st.markdown('<div class="pie">Los modelos se entrenan al iniciar la aplicación con todos los títulos disponibles.</div>', unsafe_allow_html=True)
