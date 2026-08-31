# 📊 Análisis Exploratorio y Series de Tiempo del Catálogo de Netflix

Proyecto de minería de datos desarrollado, enfocado en el análisis exploratorio del catálogo de Netflix a partir del dataset público `netflix_titles.csv` (8.807 títulos entre películas y series).

El objetivo es identificar patrones de crecimiento y composición del catálogo a lo largo del tiempo, respondiendo preguntas como:

- 📈 **Crecimiento del catálogo**: ¿cómo evolucionó la cantidad de títulos agregados a Netflix año a año (`date_added`) en comparación con su año de lanzamiento original (`release_year`)?
- 🌎 **Producción por país**: ¿qué países concentran la mayor producción de contenido disponible en la plataforma?
- 🎬 **Evolución de géneros**: ¿cómo cambiaron los géneros (`listed_in`) predominantes con el paso de los años? ¿hay tendencias hacia ciertos tipos de contenido?
- 🎞️ **Distribución por tipo y duración**: diferencias entre películas y series en cuanto a duración, clasificación (`rating`) y volumen.

## Dataset

- **Fuente**: [Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle)
- **Registros**: 8.807
- **Variables**: tipo, título, director, reparto, país, fecha de agregado, año de lanzamiento, clasificación, duración, géneros, descripción

## Tecnologías

- Python (Pandas, Matplotlib/Seaborn/Plotly)
- Jupyter Notebook

## Estructura del repositorio

```
├── data/
│   └── netflix_titles.csv
├── notebooks/
│   └── analisis_exploratorio.ipynb
├── README.md
└── requirements.txt
```

## Cómo ejecutar el proyecto

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd <nombre-del-repositorio>

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el notebook
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

## Autor

James — Ingeniería de Sistemas, UNIMINUTO Ibagué
