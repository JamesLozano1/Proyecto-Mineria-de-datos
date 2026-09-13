from src.lectorDatos import LectorDatos
from src.limpiadorDatos import LimpiadorDatos
from src.analizadorDatos import AnalizadorDatos
from src.visualizadorDatos import VisualizadorDatos


# ==========================================
# EXTRACCIÓN
# ==========================================

lector = LectorDatos("./CSV/netflix_titles.csv")
datos = lector.cargarDatos()

lectorValoraciones = LectorDatos("./CSV/netflix.csv")
datosValoraciones = lectorValoraciones.cargarDatos()

limpiadorValoraciones = LimpiadorDatos(datosValoraciones)

print("\n===== VALORES NULOS DEL CSV DE VALORACIONES =====")
print(limpiadorValoraciones.mostrarValoresNulos())

datosValoracionesLimpios = limpiadorValoraciones.limpiarDatos()

print("\n===== CSV DE VALORACIONES DESPUÉS DE LA LIMPIEZA =====")
print("Cantidad de registros:", len(datosValoracionesLimpios))


print("===== INFORMACIÓN DEL DATASET =====")

print("Cantidad de registros:", len(datos))
print("Cantidad de columnas:", len(datos.columns))


# ==========================================
# INSPECCIÓN Y LIMPIEZA
# ==========================================

limpiador = LimpiadorDatos(datos)

print("\n===== VALORES NULOS =====")
print(limpiador.mostrarValoresNulos())

datosLimpios = limpiador.limpiarDatos()
datosGeneros = limpiador.normalizarGeneros()
datosPaises = limpiador.normalizarPaises()

print("\n===== DURACIÓN PROCESADA =====")
print(
    datosLimpios[
        ["title", "type", "duration", "durationValue"]
    ].head(10)
)

print("\n===== DATOS DESPUÉS DE LA LIMPIEZA =====")
print("Cantidad de registros:", len(datosLimpios))


# ==========================================
# ANÁLISIS
# ==========================================

analizador = AnalizadorDatos(datosLimpios)
analizadorGeneros = AnalizadorDatos(datosGeneros)
analizadorPaises = AnalizadorDatos(datosPaises)
visualizador = VisualizadorDatos()
analizadorValoraciones = AnalizadorDatos(datosValoracionesLimpios)

print("\n===== PELÍCULAS Y SERIES =====")
print(analizador.analizarTipos())
visualizador.mostrarGraficaBarras(
    analizador.analizarTipos(),
    "Cantidad de películas y series en Netflix",
    "Tipo de contenido",
    "Cantidad de títulos"
)


print("\n===== TÍTULOS POR AÑO =====")
print(analizador.analizarAños())
visualizador.mostrarGraficaLinea(
    analizador.analizarTitulosAgregadosPorAño(),
    "Títulos agregados a Netflix por año",
    "Año",
    "Cantidad de títulos"
)


print("\n===== CLASIFICACIONES =====")
print(analizador.analizarClasificaciones())


print("\n===== PRINCIPALES PAÍSES =====")
print(analizadorPaises.analizarPaises())
visualizador.mostrarGraficaBarras(
    analizadorPaises.analizarPaises(),
    "Principales países productores de contenido",
    "País",
    "Cantidad de títulos"
)


print("\n===== PRINCIPALES GÉNEROS =====")
print(analizadorGeneros.analizarGeneros())
visualizador.mostrarGraficaBarras(
    analizadorGeneros.analizarGeneros(),
    "Principales géneros del catálogo",
    "Género",
    "Cantidad de títulos"
)

print("\n===== TÍTULOS AGREGADOS A NETFLIX POR AÑO =====")
print(analizador.analizarTitulosAgregadosPorAño())

print("\n===== DURACIÓN DE LAS PELÍCULAS =====")
print(analizador.analizarDuracionPeliculas())


print("\n===== DURACIÓN DE LAS SERIES =====")
print(analizador.analizarDuracionSeries())

print("\n===== VALORACIONES DE LOS USUARIOS =====")
print(analizadorValoraciones.analizarValoraciones())

print("\n===== DISTRIBUCIÓN DE VALORACIONES =====")
print(analizadorValoraciones.analizarDistribucionValoraciones())
visualizador.mostrarGraficaBarras(
    analizadorValoraciones.analizarDistribucionValoraciones(),
    "Distribución de las valoraciones de usuarios",
    "Puntuación",
    "Cantidad de títulos"
)

print("\n===== ANTIGÜEDAD DEL CONTENIDO AL SER AGREGADO =====")
print(analizador.calcularAntiguedadAlAgregar())

print("\n===== AÑO DE LANZAMIENTO Y AÑO DE INCORPORACIÓN =====")
print(analizador.analizarAñoLanzamientoYAgregado().head(10))
