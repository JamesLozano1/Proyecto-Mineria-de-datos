from src.lectorDatos import LectorDatos
from src.limpiadorDatos import LimpiadorDatos
from src.analizadorDatos import AnalizadorDatos


# ==========================================
# EXTRACCIÓN
# ==========================================

lector = LectorDatos("./CSV/netflix_titles.csv")

datos = lector.cargarDatos()


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


print("\n===== DATOS DESPUÉS DE LA LIMPIEZA =====")

print("Cantidad de registros:", len(datosLimpios))


# ==========================================
# ANÁLISIS
# ==========================================

analizador = AnalizadorDatos(datosLimpios)


print("\n===== PELÍCULAS Y SERIES =====")

print(analizador.analizarTipos())


print("\n===== TÍTULOS POR AÑO =====")

print(analizador.analizarAños())


print("\n===== CLASIFICACIONES =====")

print(analizador.analizarClasificaciones())


print("\n===== PRINCIPALES PAÍSES =====")

print(analizador.analizarPaises())


print("\n===== PRINCIPALES GÉNEROS =====")

print(analizador.analizarGeneros())

print("\n===== TÍTULOS AGREGADOS A NETFLIX POR AÑO =====")

print(analizador.analizarTitulosAgregadosPorAño())
