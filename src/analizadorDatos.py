class AnalizadorDatos:

    def __init__(self, datos):
        self.datos = datos

    def analizarTipos(self):
        resultado = self.datos["type"].value_counts()
        return resultado

    def analizarAños(self):
        resultado = self.datos["release_year"].value_counts().sort_index()
        return resultado

    def analizarClasificaciones(self):
        resultado = self.datos["rating"].value_counts()
        return resultado

    def analizarPaises(self):
        resultado = self.datos["country"].value_counts().head(10)
        return resultado

    def analizarGeneros(self):
        resultado = self.datos["listed_in"].value_counts().head(10)
        return resultado

    def analizarTitulosAgregadosPorAño(self):
        datosConFecha = self.datos.dropna(subset=["date_added"])

        resultado = (
            datosConFecha["date_added"]
            .dt.year
            .value_counts()
            .sort_index()
        )

        return resultado
