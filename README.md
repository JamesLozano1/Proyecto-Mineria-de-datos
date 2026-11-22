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
- **`ExportadorDatos`** — genera el dataset enriquecido y el resumen de análisis en CSV.
- **`ModeloArbol`** — árbol de decisión (profundidad máxima 4) que predice si un título tiene valoración alta (≥ 85) y reporta su exactitud, reglas e importancia de variables.
- **`ModeloNaiveBayes`** y **`ModeloKNN`** — los otros dos clasificadores (KNN escala las variables y elige `k` con validación interna).
- **`ComparadorModelos`** — compara los tres con los mismos datos, partición y pliegues de validación cruzada, e incluye una prueba pareada.
- **`ModeloKMeans`** — segmenta el catálogo en grupos con K-means (variables escaladas con `StandardScaler`, k elegido con silhouette), describe cada grupo y calcula la valoración promedio por grupo.
- **`main.py`** — orquesta el pipeline completo: extracción → limpieza → análisis → visualización → árbol de decisión → exportación.

## Tecnologías

- Python (Pandas, Matplotlib, scikit-learn)

## Estructura del repositorio

```
├── CSV/
│   ├── netflix_titles.csv
│   └── netflix.csv
├── src/
│   ├── lectorDatos.py
│   ├── limpiadorDatos.py
│   ├── analizadorDatos.py
│   ├── visualizadorDatos.py
│   ├── exportadorDatos.py
│   ├── modeloArbol.py
│   ├── modeloNaiveBayes.py
│   ├── modeloKNN.py
│   ├── modeloKMeans.py
│   ├── comparadorModelos.py
│   └── servicio.py          (capa de despliegue: entrena los modelos finales)
├── notebooks/            (informe, árbol, comparación de modelos y K-means, ejecutados)
├── tests/                (pruebas automáticas con pytest)
├── .github/workflows/ci.yml
├── app.py                (aplicación web en Streamlit)
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

```bash
# Clonar el repositorio
git clone https://github.com/JamesLozano1/Proyecto-Mineria-de-datos.git
cd Proyecto-Mineria-de-datos

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el pipeline completo (abre las gráficas en ventanas)
python main.py

# Alternativa sin ventanas: solo guarda los resultados
python main.py --sin-graficas
```

## Integrantes

- James Francois Lozano Gordillo
- Elian Alexis Sandoval Durán

## Institución

Corporación Universitaria Minuto de Dios (UNIMINUTO), sede Ibagué — Ingeniería de Sistemas

## Resultados generados

Al ejecutar `main.py` se crea la carpeta `CSV/resultados/` con:

- `netflix_dataset_procesado.csv` — dataset limpio con columnas calculadas.
- `resumen_analisis.csv` — resultados de todos los análisis en un solo archivo.
- `graficas/` — las gráficas del análisis en PNG.
- `arbol_decision.png` y `reglas_arbol.txt` — el árbol de decisión y sus reglas.
- `comparacion_modelos.csv` y `.png` — árbol vs. Naive Bayes vs. KNN.
- `kmeans_perfiles.csv`, `titulos_con_grupo.csv` y `kmeans_*.png` — grupos de K-means, su perfil y la elección de k.


## Despliegue (fase de CRISP-DM)

El proyecto se despliega como una **aplicación web en Streamlit** (`app.py`) con cuatro pestañas:
predicción de un título (Naive Bayes predice y el árbol explica paso a paso), análisis del catálogo,
grupos de K-means y comparación de modelos. La capa `src/servicio.py` carga y limpia los datos y
entrena los modelos finales con todos los títulos al iniciar.

```
GitHub (código + CSV)  →  CI (pytest + pipeline)  →  Streamlit Community Cloud / Docker  →  URL pública
```

### 1. Probar en local
```bash
pip install -r requirements.txt
streamlit run app.py          # http://localhost:8501
pytest -q                     # 6 pruebas automáticas
```

### 2. Subir a GitHub
```bash
git init
git add .
git commit -m "Proyecto de minería de datos: pipeline, modelos y aplicación web"
git branch -M main
git remote add origin https://github.com/JamesLozano1/Proyecto-Mineria-de-datos.git
git push -u origin main
```
Si el repositorio ya tiene contenido, primero se clona y se copian estos archivos encima. En cada `push`, GitHub Actions
(`.github/workflows/ci.yml`) instala las dependencias, corre las pruebas y ejecuta el pipeline completo.

### 3. Publicar en producción (Streamlit Community Cloud, gratis)
1. Entrar a https://share.streamlit.io con la cuenta de GitHub.
2. **Create app** → elegir el repositorio, la rama `main` y el archivo principal `app.py`.
3. **Deploy**. La plataforma instala `requirements.txt` y entrega una URL pública (`https://<nombre>.streamlit.app`).
4. Cada `git push` a `main` actualiza la aplicación automáticamente.

### Alternativa: Docker
```bash
docker build -t netflix-app .
docker run -p 8501:8501 netflix-app
```

> Los modelos se entrenan con solo 137 títulos y aciertan alrededor de 60 % (adivinar la clase más común da 54 %).
> La aplicación lo advierte en pantalla: es una orientación, no una predicción confiable.
