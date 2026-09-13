import pandas as pd


class AnalizadorDatos:

    def __init__(self, datos):
        self.datos = datos

    def analizarTipos(self):
        resultado = self.datos["type"].value_counts()
        return resultado

    def analizarAños(self):
        resultado = (
            self.datos["release_year"]
            .value_counts()
            .sort_index()
        )
        return resultado

    def analizarClasificaciones(self):
        resultado = self.datos["rating"].value_counts()
        return resultado

    def analizarPaises(self):
        resultado = (
            self.datos["country"]
            .value_counts()
            .head(10)
        )
        return resultado

    def analizarGeneros(self):
        resultado = (
            self.datos["listed_in"]
            .value_counts()
            .head(10)
        )
        return resultado

    def analizarTitulosAgregadosPorAño(self):
        datosConFecha = self.datos.dropna(
            subset=["date_added"]
        )
        resultado = (
            datosConFecha["date_added"]
            .dt.year
            .value_counts()
            .sort_index()
        )
        return resultado

    def analizarDuracionPeliculas(self):
        peliculas = self.datos[
            self.datos["type"] == "Movie"
        ]
        resultado = peliculas["durationValue"].describe()
        return resultado

    def analizarDuracionSeries(self):
        series = self.datos[
            self.datos["type"] == "TV Show"
        ]
        resultado = series["durationValue"].describe()
        return resultado

    def analizarValoraciones(self):
        datosConValoracion = self.datos.dropna(
            subset=["user_rating_score"]
        )
        resultado = datosConValoracion["user_rating_score"].describe()
        return resultado

    def analizarDistribucionValoraciones(self):
        datosConValoracion = self.datos.dropna(
            subset=["user_rating_score"]
        )
        resultado = (
            datosConValoracion["user_rating_score"]
            .value_counts()
            .sort_index()
        )
        return resultado

    def analizarValoracionPorGenero(self, datosPrincipal, minimoTitulos=5):
        """
        Cruza este dataset de valoraciones (self.datos, debe tener
        "title" y "user_rating_score") con el dataset principal
        (datosPrincipal, debe tener "title" y "listed_in" sin explotar)
        para calcular la valoración promedio de usuarios por género.

        minimoTitulos: un género solo se incluye en el resultado si tiene
        al menos esa cantidad de títulos valorados, para evitar promedios
        poco representativos calculados sobre 1 o 2 títulos.
        """
        datosConValoracion = self.datos.dropna(
            subset=["user_rating_score"]
        )

        principalUnico = datosPrincipal.drop_duplicates(
            subset=["title"]
        )[["title", "listed_in"]]

        combinado = pd.merge(
            datosConValoracion, principalUnico, on="title", how="inner"
        )

        combinado["listed_in"] = combinado["listed_in"].str.split(", ")
        combinado = combinado.explode("listed_in")
        combinado["listed_in"] = combinado["listed_in"].str.strip()

        resumen = combinado.groupby("listed_in")["user_rating_score"].agg(
            ["count", "mean"]
        )
        resumen = resumen[resumen["count"] >= minimoTitulos]
        resumen = resumen.sort_values("mean", ascending=False)

        return resumen["mean"]

    def obtenerPromedioDuracion(self):
        peliculas = self.datos[
            self.datos["type"] == "Movie"
        ]
        series = self.datos[
            self.datos["type"] == "TV Show"
        ]
        promedioPeliculas = peliculas["durationValue"].mean()
        promedioSeries = series["durationValue"].mean()
        resultado = {
            "Películas": promedioPeliculas,
            "Series": promedioSeries
        }
        return resultado

    def analizarAñoLanzamientoYAgregado(self):
        datosConFecha = self.datos.dropna(
            subset=["date_added"]
        )
        resultado = datosConFecha[
            ["release_year", "date_added"]
        ].copy()
        resultado["added_year"] = (
            resultado["date_added"].dt.year
        )
        return resultado

    def calcularAntiguedadAlAgregar(self):
        datosConFecha = self.datos.dropna(
            subset=["date_added"]
        ).copy()
        datosConFecha["added_year"] = (
            datosConFecha["date_added"].dt.year
        )
        datosConFecha["antiguedad"] = (
            datosConFecha["added_year"]
            - datosConFecha["release_year"]
        )
        return datosConFecha["antiguedad"].describe()

    def obtenerDatosEnriquecidos(self):
        """
        Devuelve el dataset completo con todas las columnas calculadas
        durante el análisis (año de incorporación a Netflix y antigüedad
        al agregar), listo para exportarse como dataset final del proyecto.

        No modifica self.datos: trabaja sobre una copia.
        """
        datosEnriquecidos = self.datos.copy()

        if "date_added" in datosEnriquecidos.columns:
            datosEnriquecidos["added_year"] = (
                datosEnriquecidos["date_added"].dt.year
            )

        if "added_year" in datosEnriquecidos.columns and "release_year" in datosEnriquecidos.columns:
            datosEnriquecidos["antiguedad_al_agregar"] = (
                datosEnriquecidos["added_year"]
                - datosEnriquecidos["release_year"]
            )

        return datosEnriquecidos
