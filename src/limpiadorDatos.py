import pandas as pd


class LimpiadorDatos:

    def __init__(self, datos):
        self.datos = datos.copy()

    def mostrarValoresNulos(self):
        return self.datos.isnull().sum()

    def eliminarDuplicados(self):
        cantidadAntes = len(self.datos)
        self.datos = self.datos.drop_duplicates().copy()
        cantidadDespues = len(self.datos)
        duplicadosEliminados = cantidadAntes - cantidadDespues
        print("Duplicados eliminados:", duplicadosEliminados)
        return self.datos

    def reemplazarValoresNulos(self):
        columnasTexto = ["director", "cast", "country", "rating", "duration"]
        for columna in columnasTexto:
            if columna in self.datos.columns:
                self.datos[columna] = self.datos[columna].fillna("Desconocido")
        return self.datos

    def convertirFecha(self):
        if "date_added" in self.datos.columns:
            # Asignación directa: esto SÍ cambia el dtype a datetime64 correctamente.
            # (usar .loc aquí rompe la conversión, ver explicación arriba)
            self.datos["date_added"] = pd.to_datetime(
                self.datos["date_added"], errors="coerce"
            )
        return self.datos

    def normalizarGeneros(self):
        datos = self.datos.copy()
        if "listed_in" in datos.columns:
            datos["listed_in"] = datos["listed_in"].str.split(", ")
            datos = datos.explode("listed_in")
            datos["listed_in"] = datos["listed_in"].str.strip()
        return datos

    def normalizarPaises(self):
        datos = self.datos.copy()
        if "country" in datos.columns:
            datos["country"] = datos["country"].str.split(", ")
            datos = datos.explode("country")
            datos["country"] = datos["country"].str.strip()
        return datos

    def procesarDuracion(self):
        if "duration" in self.datos.columns:
            self.datos["durationValue"] = (
                self.datos["duration"].str.split(" ").str[0]
            )
            self.datos["durationValue"] = pd.to_numeric(
                self.datos["durationValue"], errors="coerce"
            )
        return self.datos

    def mostrarRatingsSospechosos(self):
        if "rating" not in self.datos.columns:
            return pd.DataFrame()

        ratingsValidos = [
            "G", "PG", "PG-13", "R", "NC-17",
            "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG",
            "TV-14", "TV-MA", "NR", "UR"
        ]

        ratingsSospechosos = self.datos[
            (~self.datos["rating"].isin(ratingsValidos)) &
            (self.datos["rating"] != "Desconocido")
        ]

        columnas = [c for c in ["title", "rating",
                                "duration", "type"] if c in self.datos.columns]
        return ratingsSospechosos[columnas]

    def corregirRatingsSospechosos(self):
        if "rating" not in self.datos.columns or "duration" not in self.datos.columns:
            return self.datos

        ratingsValidos = [
            "G", "PG", "PG-13", "R", "NC-17",
            "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG",
            "TV-14", "TV-MA", "NR", "UR"
        ]

        filasSospechosas = (
            (~self.datos["rating"].isin(ratingsValidos)) &
            (self.datos["rating"] != "Desconocido")
        )

        self.datos.loc[filasSospechosas,
                       "duration"] = self.datos.loc[filasSospechosas, "rating"]
        self.datos.loc[filasSospechosas, "rating"] = "Desconocido"

        return self.datos

    def limpiarDatos(self):
        self.eliminarDuplicados()
        self.reemplazarValoresNulos()
        self.convertirFecha()
        return self.datos

    def corregirYProcesar(self):
        self.corregirRatingsSospechosos()
        self.procesarDuracion()
        return self.datos
