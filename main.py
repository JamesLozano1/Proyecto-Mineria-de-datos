from src.lectorDatos import LectorDatos
from src.limpiadorDatos import LimpiadorDatos
from src.analizadorDatos import AnalizadorDatos
from src.visualizadorDatos import VisualizadorDatos
from src.exportadorDatos import ExportadorDatos


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

visualizador = VisualizadorDatos()


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
