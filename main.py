import sys

SIN_GRAFICAS = "--sin-graficas" in sys.argv   # python main.py --sin-graficas: no abre ventanas, solo guarda los PNG
if SIN_GRAFICAS:
    import matplotlib
    matplotlib.use("Agg")

from src.lectorDatos import LectorDatos
from src.limpiadorDatos import LimpiadorDatos
from src.analizadorDatos import AnalizadorDatos
from src.visualizadorDatos import VisualizadorDatos
from src.exportadorDatos import ExportadorDatos
from src.modeloArbol import ModeloArbol
from src.modeloNaiveBayes import ModeloNaiveBayes
from src.modeloKNN import ModeloKNN
from src.comparadorModelos import ComparadorModelos
from src.modeloKMeans import ModeloKMeans


# ==========================================
# EXTRACCIÓN DE DATOS
# ==========================================

lector = LectorDatos("./CSV/netflix_titles.csv")

datos = lector.cargarDatos()


lectorValoraciones = LectorDatos("./CSV/netflix.csv")

datosValoraciones = lectorValoraciones.cargarDatos()


# ==========================================
# INFORMACIÓN INICIAL
# ==========================================

print("==========================================")
print("      PROYECTO MINERÍA DE DATOS")
print("      ANÁLISIS DEL CATÁLOGO NETFLIX")
print("==========================================")


print("\n===== DATASET PRINCIPAL =====")

print("Cantidad de registros:", len(datos))
print("Cantidad de columnas:", len(datos.columns))


print("\n===== DATASET DE VALORACIONES =====")

print("Cantidad de registros:", len(datosValoraciones))
print("Cantidad de columnas:", len(datosValoraciones.columns))


# ==========================================
# LIMPIEZA DATASET PRINCIPAL
# ==========================================

limpiador = LimpiadorDatos(datos)

print("\n===== VALORES NULOS DATASET PRINCIPAL =====")
print(limpiador.mostrarValoresNulos())

datosLimpios = limpiador.limpiarDatos()  # duplicados + nulos + fechas

print("\n===== RATINGS SOSPECHOSOS DETECTADOS =====")
print(limpiador.mostrarRatingsSospechosos())  # ahora sí muestra resultados

datosLimpios = limpiador.corregirYProcesar()  # corrige y calcula durationValue

print("\n===== DATOS PRINCIPALES DESPUÉS DE LA LIMPIEZA =====")
print("Cantidad de registros:", len(datosLimpios))


# ==========================================
# LIMPIEZA DATASET DE VALORACIONES
# ==========================================

limpiadorValoraciones = LimpiadorDatos(datosValoraciones)


print("\n===== VALORES NULOS DATASET VALORACIONES =====")

print(limpiadorValoraciones.mostrarValoresNulos())


datosValoracionesLimpios = (
    limpiadorValoraciones.limpiarDatos()
)


print("\n===== VALORACIONES DESPUÉS DE LA LIMPIEZA =====")

print(
    "Cantidad de registros:",
    len(datosValoracionesLimpios)
)


# ==========================================
# PREPARACIÓN DE GÉNEROS Y PAÍSES
# ==========================================

datosGeneros = limpiador.normalizarGeneros()

datosPaises = limpiador.normalizarPaises()


# ==========================================
# CREACIÓN DE ANALIZADORES
# ==========================================

analizador = AnalizadorDatos(datosLimpios)

analizadorGeneros = AnalizadorDatos(datosGeneros)

analizadorPaises = AnalizadorDatos(datosPaises)

analizadorValoraciones = AnalizadorDatos(
    datosValoracionesLimpios
)


# ==========================================
# VISUALIZADOR
# ==========================================

visualizador = VisualizadorDatos(
    mostrar=not SIN_GRAFICAS,
    carpeta="./CSV/resultados/graficas"
)


# ==========================================
# ANÁLISIS
# ==========================================

print("\n==========================================")
print("              ANÁLISIS")
print("==========================================")


print("\n===== PELÍCULAS Y SERIES =====")

print(
    analizador.analizarTipos()
)


print("\n===== TÍTULOS POR AÑO DE LANZAMIENTO =====")

print(
    analizador.analizarAños()
)


print("\n===== CLASIFICACIONES =====")

print(
    analizador.analizarClasificaciones()
)


print("\n===== PRINCIPALES PAÍSES =====")

print(
    analizadorPaises.analizarPaises()
)


print("\n===== PRINCIPALES GÉNEROS =====")

print(
    analizadorGeneros.analizarGeneros()
)


print("\n===== TÍTULOS AGREGADOS A NETFLIX POR AÑO =====")

print(
    analizador.analizarTitulosAgregadosPorAño()
)


print("\n===== DURACIÓN DE LAS PELÍCULAS =====")

print(
    analizador.analizarDuracionPeliculas()
)


print("\n===== DURACIÓN DE LAS SERIES =====")

print(
    analizador.analizarDuracionSeries()
)


print("\n===== ANTIGÜEDAD DEL CONTENIDO AL SER AGREGADO =====")

print(
    analizador.calcularAntiguedadAlAgregar()
)


print("\n===== VALORACIONES DE LOS USUARIOS =====")

print(
    analizadorValoraciones.analizarValoraciones()
)


# ==========================================
# VISUALIZACIONES
# ==========================================

print("\n==========================================")
print("          GENERANDO GRÁFICAS")
print("==========================================")


visualizador.mostrarGraficaBarras(
    analizador.analizarTipos(),
    "Cantidad de películas y series en Netflix",
    "Tipo de contenido",
    "Cantidad de títulos"
)


visualizador.mostrarGraficaLinea(
    analizador.analizarTitulosAgregadosPorAño(),
    "Títulos agregados a Netflix por año",
    "Año",
    "Cantidad de títulos"
)


visualizador.mostrarGraficaBarras(
    analizadorPaises.analizarPaises(),
    "Principales países productores de contenido",
    "País",
    "Cantidad de títulos"
)


visualizador.mostrarGraficaBarras(
    analizadorGeneros.analizarGeneros(),
    "Principales géneros del catálogo",
    "Género",
    "Cantidad de títulos"
)


visualizador.mostrarGraficaBarras(
    analizadorValoraciones.analizarValoracionPorGenero(datosLimpios),
    "Valoración promedio de usuarios por género",
    "Género",
    "Puntuación promedio"
)


# ==========================================
# ÁRBOL DE DECISIÓN (profundidad máxima 4)
# ==========================================

print("\n==========================================")
print("        ÁRBOL DE DECISIÓN")
print("==========================================")

modeloArbol = ModeloArbol(maxProfundidad=4, umbral=85)
modeloArbol.prepararDatos(datosValoracionesLimpios, datosLimpios)
modeloArbol.entrenar()
metricas = modeloArbol.evaluar()

print("Títulos usados:", metricas["titulos"], "| Variables:", metricas["variables"])
print("Profundidad:", metricas["profundidad"], "| Hojas:", metricas["hojas"])
print("Exactitud en entrenamiento: %.3f" % metricas["exactitud_entrenamiento"])
print("Exactitud en prueba:        %.3f (línea base %.3f)" % (metricas["exactitud_prueba"], metricas["linea_base_prueba"]))
print("Validación cruzada:         %.3f +- %.3f (línea base %.3f)" % (
    metricas["validacion_cruzada"], metricas["validacion_cruzada_desv"], metricas["linea_base"]))
print("\nVariables más importantes:")
print(modeloArbol.importancias().head(6).round(3))
print("\nReglas del árbol:")
print(modeloArbol.reglas())

modeloArbol.guardarGrafica("./CSV/resultados/arbol_decision.png", mostrar=not SIN_GRAFICAS)
with open("./CSV/resultados/reglas_arbol.txt", "w", encoding="utf-8") as archivo:
    archivo.write(modeloArbol.reglas())


# ==========================================
# COMPARACIÓN: ÁRBOL vs NAIVE BAYES vs KNN
# ==========================================

print("\n==========================================")
print("   COMPARACIÓN: ÁRBOL vs NAIVE BAYES vs KNN")
print("==========================================")

comparador = ComparadorModelos(modeloArbol.X, modeloArbol.y)
comparador.agregar("Árbol de decisión", modeloArbol.estimador())
comparador.agregar("Naive Bayes", ModeloNaiveBayes().estimador())
comparador.agregar("KNN", ModeloKNN().estimador())   # escala las variables y elige k con validación interna

tablaModelos = comparador.comparar()
print(tablaModelos.round(3).to_string())
print("\nLínea base (adivinar la clase más común): %.3f" % comparador.lineaBase)

pareada = comparador.pruebaPareada("Árbol de decisión")
print("\nDiferencia frente al árbol (prueba pareada):")
print(pareada.round(3).to_string())

print("\nMayor exactitud en validación cruzada:", comparador.mejor())
if (pareada["p_valor"] > 0.05).all():
    print("Ninguna diferencia frente al árbol es estadísticamente significativa (p > 0,05).")

tablaModelos.round(4).to_csv("./CSV/resultados/comparacion_modelos.csv")
comparador.guardarGrafica("./CSV/resultados/comparacion_modelos.png", mostrar=not SIN_GRAFICAS)


# ==========================================
# K-MEANS: SEGMENTACIÓN DEL CATÁLOGO
# ==========================================

print("\n==========================================")
print("   K-MEANS: SEGMENTACIÓN DEL CATÁLOGO")
print("==========================================")

modeloKMeans = ModeloKMeans()
modeloKMeans.prepararDatos(datosLimpios)
print("Títulos:", len(modeloKMeans.X), "| Variables (escaladas con StandardScaler):", len(modeloKMeans.variables))
print(modeloKMeans.evaluarK(2, 12).round(3).to_string())

modeloKMeans.entrenar()
print("\nk elegido:", modeloKMeans.k, "(menor k a 0,01 del silhouette máximo)")

perfiles = modeloKMeans.perfiles()
print("\nPerfil de cada grupo:")
print(perfiles[["nombre", "titulos", "porcentaje", "pct_series", "anio_medio", "pct_infantil", "pct_adulto"]].round(1).to_string())

valoracionGrupos = modeloKMeans.valoracionPorGrupo(datosValoracionesLimpios)
print("\nValoración de usuarios por grupo (solo grupos con 5 o más títulos valorados son confiables):")
print(valoracionGrupos.round(1).to_string())

perfiles.round(2).to_csv("./CSV/resultados/kmeans_perfiles.csv")
modeloKMeans.titulos[["show_id", "title", "grupo"]].to_csv("./CSV/resultados/titulos_con_grupo.csv", index=False)
modeloKMeans.guardarGraficaK("./CSV/resultados/kmeans_eleccion_k.png", mostrar=not SIN_GRAFICAS)
modeloKMeans.guardarGraficaGrupos("./CSV/resultados/kmeans_grupos.png", mostrar=not SIN_GRAFICAS)
modeloKMeans.guardarGraficaCentroides("./CSV/resultados/kmeans_centroides.png", mostrar=not SIN_GRAFICAS)


# ==========================================
# EXPORTACIÓN DE DATASETS FINALES
# ==========================================

print("\n==========================================")
print("       EXPORTANDO DATASETS FINALES")
print("==========================================")

exportador = ExportadorDatos("./CSV/resultados")

# --- Dataset enriquecido: dataset principal + columnas calculadas ---
# (durationValue, added_year, antiguedad_al_agregar)

datosEnriquecidos = analizador.obtenerDatosEnriquecidos()

exportador.exportarDatosEnriquecidos(
    datosEnriquecidos,
    "netflix_dataset_procesado.csv"
)

# --- Dataset resumen: resultados de todos los análisis en un solo archivo ---

resumenAnalisis = {
    "Tipos de contenido": analizador.analizarTipos(),
    "Títulos por año de lanzamiento": analizador.analizarAños(),
    "Clasificaciones": analizador.analizarClasificaciones(),
    "Principales países": analizadorPaises.analizarPaises(),
    "Principales géneros": analizadorGeneros.analizarGeneros(),
    "Títulos agregados a Netflix por año": analizador.analizarTitulosAgregadosPorAño(),
    "Duración de las películas": analizador.analizarDuracionPeliculas(),
    "Duración de las series": analizador.analizarDuracionSeries(),
    "Antigüedad al agregar": analizador.calcularAntiguedadAlAgregar(),
    "Valoraciones de usuarios": analizadorValoraciones.analizarValoraciones(),
    "Valoración por género": analizadorValoraciones.analizarValoracionPorGenero(datosLimpios),
}

exportador.exportarResumenAnalisis(
    resumenAnalisis,
    "resumen_analisis.csv"
)


print("\nProceso terminado. Resultados en ./CSV/resultados/")
