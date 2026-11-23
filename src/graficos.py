import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

ROJO, GRIS, CLARO, AZUL, NARANJA = "#E50914", "#8C8C8C", "#F5F2EC", "#4DA3FF", "#FF9F43"
TEXTO, REJILLA = "#D9D2C7", "#2E2E2E"


def _estilo(grafica, alto=300):
    """Aspecto común (fondo transparente, ejes claros) para que combine con el tema oscuro."""
    return (grafica.properties(height=alto)
            .configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelColor=TEXTO, titleColor=TEXTO, gridColor=REJILLA, domainColor="#4A4540",
                            tickColor="#4A4540", labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=TEXTO, titleColor=TEXTO, labelFontSize=12)
            .configure_title(color=CLARO, fontSize=16, anchor="start"))


def donut_tipo(catalogo):
    t = catalogo.drop_duplicates("title")["type"].map({"Movie": "Películas", "TV Show": "Series"}).value_counts()
    df = pd.DataFrame({"tipo": t.index, "titulos": t.values})
    df["porcentaje"] = (df["titulos"] / df["titulos"].sum()).round(3)
    g = alt.Chart(df, title="Películas y series").mark_arc(innerRadius=70, outerRadius=115).encode(
        theta="titulos:Q",
        color=alt.Color("tipo:N", scale=alt.Scale(domain=["Películas", "Series"], range=[ROJO, GRIS]), title=None),
        tooltip=["tipo", "titulos", alt.Tooltip("porcentaje:Q", format=".1%")])
    return _estilo(g, 260)


def area_anual(catalogo):
    c = catalogo.drop_duplicates("title").dropna(subset=["date_added"])
    df = c.groupby(c["date_added"].dt.year).size().rename_axis("anio").reset_index(name="titulos")
    df = df[df["anio"] >= 2008]
    base = alt.Chart(df, title="Títulos agregados a Netflix por año").encode(
        x=alt.X("anio:Q", title="Año", axis=alt.Axis(format="d")), y=alt.Y("titulos:Q", title="Títulos"),
        tooltip=[alt.Tooltip("anio:Q", title="Año", format="d"), alt.Tooltip("titulos:Q", title="Títulos", format=",d")])
    return _estilo(base.mark_area(color=ROJO, opacity=0.3) + base.mark_line(color=ROJO, point=alt.OverlayMarkDef(color=ROJO)), 260)


def barras_h(serie, titulo, color=ROJO, formato=",d", alto=300):
    df = pd.DataFrame({"nombre": serie.index.astype(str), "valor": serie.values})
    base = alt.Chart(df, title=titulo).encode(
        y=alt.Y("nombre:N", sort="-x", title=None), x=alt.X("valor:Q", title=None),
        tooltip=["nombre", alt.Tooltip("valor:Q", format=formato)])
    barras = base.mark_bar(color=color, cornerRadiusEnd=4)
    texto = base.mark_text(align="left", dx=4, color=CLARO, fontSize=12).encode(text=alt.Text("valor:Q", format=formato))
    return _estilo(barras + texto, alto)


def hist_duracion(catalogo):
    c = catalogo.drop_duplicates("title")
    m = c.loc[c["type"] == "Movie", "durationValue"].dropna()
    conteos, bordes = np.histogram(m, bins=30)          # se agrupa antes de graficar (pocos datos al navegador)
    df = pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "peliculas": conteos})
    barras = alt.Chart(df, title="Duración de las películas").mark_bar(color=ROJO, opacity=0.9).encode(
        x=alt.X("desde:Q", title="Minutos"), x2="hasta:Q", y=alt.Y("peliculas:Q", title="Películas"), y2=alt.datum(0),
        tooltip=[alt.Tooltip("desde:Q", title="Desde (min)", format=".0f"), alt.Tooltip("hasta:Q", title="Hasta (min)", format=".0f"), "peliculas"])
    regla = alt.Chart(pd.DataFrame({"m": [m.median()]})).mark_rule(color=CLARO, strokeDash=[5, 4], size=2).encode(
        x="m:Q", tooltip=[alt.Tooltip("m:Q", title="Mediana (min)", format=".0f")])
    return _estilo(barras + regla, 260)


def valoracion_genero(df, promedio_general):
    df = df.copy()
    niveles = ["85 o más", "80 a 85", "menos de 80"]
    df["nivel"] = np.where(df["promedio"] >= 85, niveles[0], np.where(df["promedio"] >= 80, niveles[1], niveles[2]))
    base = alt.Chart(df, title="Valoración promedio por género (3 o más títulos valorados)").encode(
        y=alt.Y("genero:N", sort="-x", title=None),
        x=alt.X("promedio:Q", title="Valoración promedio (sobre 100)", scale=alt.Scale(domain=[65, 100])),
        tooltip=["genero", alt.Tooltip("promedio:Q", format=".1f"), alt.Tooltip("mediana:Q", format=".1f"), "titulos"])
    puntos = base.mark_circle(size=170, opacity=1).encode(
        color=alt.Color("nivel:N", scale=alt.Scale(domain=niveles, range=[ROJO, "#C9C2B8", "#6B645E"]), title="Valoración"))
    texto = base.mark_text(align="left", dx=12, color=CLARO, fontSize=12).encode(text=alt.Text("promedio:Q", format=".1f"))
    regla = alt.Chart(pd.DataFrame({"x": [promedio_general]})).mark_rule(color=AZUL, strokeDash=[5, 4], size=2).encode(
        x="x:Q", tooltip=[alt.Tooltip("x:Q", title="Promedio general", format=".1f")])
    return _estilo(puntos + texto + regla, 28 * max(len(df), 4) + 40)


def box_audiencia(val):
    orden = ["Infantil/Familiar", "Otra (PG, TV-PG, UR)", "Adolescente/Adulto"]
    g = alt.Chart(val, title="Valoración según el público").mark_boxplot(size=50, extent="min-max").encode(
        x=alt.X("audiencia:N", sort=orden, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("user_rating_score:Q", title="Valoración", scale=alt.Scale(domain=[50, 100])),
        color=alt.Color("audiencia:N", sort=orden, scale=alt.Scale(domain=orden, range=["#6B645E", "#C9C2B8", ROJO]), legend=None))
    return _estilo(g, 320)


def dispersion_grupos(df):
    seleccion = alt.selection_point(fields=["grupo"], bind="legend")
    g = alt.Chart(df, title="Los 10 grupos de títulos, vistos en 2 dimensiones").mark_circle(size=22).encode(
        x=alt.X("x:Q", title="Componente 1"), y=alt.Y("y:Q", title="Componente 2"),
        color=alt.Color("grupo:N", scale=alt.Scale(scheme="tableau10"), title="Grupo"),
        opacity=alt.condition(seleccion, alt.value(0.7), alt.value(0.08)),
        tooltip=["title", "nombre", "grupo"]).add_params(seleccion).interactive()
    return _estilo(g, 420)


def heatmap_centroides(df):
    base = alt.Chart(df, title="Perfil de cada grupo (rojo: por encima del promedio, azul: por debajo)").encode(
        x=alt.X("variable:N", title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y("grupo:N", title=None, sort=None))
    celdas = base.mark_rect().encode(color=alt.Color("valor:Q", scale=alt.Scale(scheme="redblue", domain=[-2, 2], reverse=True),
                                                      legend=alt.Legend(title="Desviaciones")),
                                     tooltip=["grupo", "variable", alt.Tooltip("valor:Q", format=".2f")])
    texto = base.mark_text(fontSize=10).encode(text=alt.Text("valor:Q", format=".1f"),
                                               color=alt.condition("abs(datum.valor) > 1.5", alt.value("white"), alt.value("#222")))
    return _estilo(celdas + texto, 380)


def sensibilidad(df, x_actual, etiqueta_x):
    lineas = alt.Chart(df, title="¿Cómo cambia la probabilidad si varía %s?" % etiqueta_x.lower()).mark_line(point=True, size=3).encode(
        x=alt.X("x:Q", title=etiqueta_x), y=alt.Y("probabilidad:Q", title="Probabilidad de valoración alta", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("modelo:N", scale=alt.Scale(domain=["Naive Bayes", "Árbol de decisión"], range=[ROJO, AZUL]), title=None),
        tooltip=["modelo", alt.Tooltip("x:Q", title=etiqueta_x), alt.Tooltip("probabilidad:Q", format=".0%")])
    regla = alt.Chart(pd.DataFrame({"x": [x_actual]})).mark_rule(color=CLARO, strokeDash=[5, 4], size=2).encode(x="x:Q")
    return _estilo(lineas + regla, 280)


def comparacion(tabla, columna, titulo, base=None, formato=".1%", escala=None):
    df = tabla.reset_index()
    ejey = alt.Y("%s:Q" % columna, title=None, axis=alt.Axis(format=formato), scale=alt.Scale(domain=escala) if escala else alt.Undefined)
    barras = alt.Chart(df, title=titulo).mark_bar(cornerRadiusEnd=4, size=60).encode(
        x=alt.X("modelo:N", title=None, sort=["Árbol de decisión", "Naive Bayes", "KNN"], axis=alt.Axis(labelAngle=0)), y=ejey,
        color=alt.Color("modelo:N", scale=alt.Scale(domain=["Árbol de decisión", "Naive Bayes", "KNN"], range=[ROJO, CLARO, GRIS]), legend=None),
        tooltip=["modelo", alt.Tooltip("%s:Q" % columna, format=formato)])
    texto = barras.mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(text=alt.Text("%s:Q" % columna, format=formato), color=alt.value(CLARO))
    capas = barras + texto
    if base is not None:
        capas = capas + alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(color=AZUL, strokeDash=[5, 4], size=2).encode(y="y:Q")
    return _estilo(capas, 280)
