# 📊 Análisis Exploratorio y Series de Tiempo del Catálogo de Netflix

Proyecto de minería de datos desarrollado bajo la metodología **CRISP-DM**, enfocado en el análisis exploratorio del catálogo de Netflix y de las valoraciones de usuarios asociadas a su contenido.

Se trabaja sobre dos fuentes de datos:

- `netflix_titles.csv` — catálogo completo de Netflix (8.807 títulos entre películas y series).
- `netflix.csv` — valoraciones de usuarios sobre parte de ese contenido (puntuación, clasificación y descripción de la clasificación).

El objetivo es identificar patrones de crecimiento y composición del catálogo a lo largo del tiempo, y complementar ese análisis con la percepción de los usuarios, respondiendo preguntas como:

- 📈 **Crecimiento del catálogo**: ¿cómo evolucionó la cantidad de títulos agregados a Netflix año a año (`date_added`) en comparación con su año de lanzamiento original (`release_year`)?
- 🌎 **Producción por país**: ¿qué países concentran la mayor producción de contenido disponible en la plataforma?
- 🎬 **Evolución de géneros**: ¿cómo cambiaron los géneros (`listed_in`) predominantes con el paso de los años? ¿hay tendencias hacia ciertos tipos de contenido?
- 🎞️ **Distribución por tipo y duración**: diferencias entre películas y series en cuanto a duración, clasificación (`rating`) y volumen.
- ⭐ **Valoraciones de usuarios**: ¿cómo se distribuyen las puntuaciones de los usuarios sobre el contenido?
- ⏳ **Antigüedad al agregar**: ¿cuánto tiempo pasa, en promedio, entre el año de lanzamiento de un título y su incorporación a Netflix?

## Metodología

El proyecto sigue las seis fases de CRISP-DM:

1. **Comprensión del negocio** – definir el problema y las preguntas de interés sobre el catálogo y las valoraciones.
2. **Comprensión de los datos** – exploración inicial, calidad de datos y valores nulos (`director`, `cast`, `country`, `user_rating_score`).
3. **Preparación de los datos** – eliminación de duplicados, limpieza de nulos, conversión de fechas, normalización de géneros y países, extracción del valor numérico de duración.
4. **Modelado** – análisis estadístico descriptivo, series de tiempo y visualizaciones (barras, líneas).
5. **Evaluación** – validación de los hallazgos frente a las preguntas de negocio planteadas.
6. **Despliegue** – presentación de resultados mediante gráficas y conclusiones.

## Datasets

| Archivo | Fuente | Registros | Descripción |
|---|---|---|---|
| `netflix_titles.csv` | [Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle) | 8.807 | Tipo, título, director, reparto, país, fecha de agregado, año de lanzamiento, clasificación, duración, géneros, descripción |
| `netflix.csv` | Dataset de valoraciones de usuarios | 1.000 | Título, clasificación, descripción de la clasificación, nivel de clasificación, año de lanzamiento, puntuación y tamaño de valoración de usuarios |

## Arquitectura del proyecto

El pipeline está construido en Python con clases separadas por responsabilidad:

- **`LectorDatos`** — carga los archivos CSV con Pandas.
- **`LimpiadorDatos`** — elimina duplicados, reemplaza nulos, convierte fechas, normaliza géneros/países y procesa la duración.
- **`AnalizadorDatos`** — calcula los conteos, promedios y estadísticas descriptivas para cada pregunta del proyecto.
- **`VisualizadorDatos`** — genera las gráficas de barras y de línea a partir de los resultados del análisis.
- **`main.py`** — orquesta el pipeline completo: extracción → limpieza → análisis → visualización.

## Tecnologías

- Python (Pandas, Matplotlib)

## Estructura del repositorio

```
├── CSV/
│   ├── netflix_titles.csv
│   └── netflix.csv
├── src/
│   ├── lectorDatos.py
│   ├── limpiadorDatos.py
│   ├── analizadorDatos.py
│   └── visualizadorDatos.py
├── main.py
└── README.md
```

## Cómo ejecutar el proyecto

```bash
# Clonar el repositorio
git clone https://github.com/JamesLozano1/Proyecto-Mineria-de-datos.git
cd Proyecto-Mineria-de-datos

# Instalar dependencias
pip install pandas matplotlib

# Ejecutar el pipeline
python main.py
```

## Integrantes

- James Francois Lozano Gordillo
- Elian Alexis Sandoval Durán

## Institución

Corporación Universitaria Minuto de Dios (UNIMINUTO), sede Ibagué — Ingeniería de Sistemas