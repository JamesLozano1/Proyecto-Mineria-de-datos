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
