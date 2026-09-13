import pandas as pd


class LimpiadorDatos:

    def __init__(self, datos):
        self.datos = datos

    def mostrarValoresNulos(self):
        return self.datos.isnull().sum()

    def eliminarDuplicados(self):
        cantidadAntes = len(self.datos)
        self.datos = self.datos.drop_duplicates()
        cantidadDespues = len(self.datos)
        duplicadosEliminados = cantidadAntes - cantidadDespues
        print(
            "Duplicados eliminados:",
            duplicadosEliminados
        )
        return self.datos

    def reemplazarValoresNulos(self):
        columnasTexto = [
            "director",
            "cast",
            "country",
            "rating",
            "duration"
        ]
        for columna in columnasTexto:
            if columna in self.datos.columns:
                self.datos[columna] = self.datos[columna].fillna(
                    "Desconocido"
                )
        return self.datos

    def convertirFecha(self):
        if "date_added" in self.datos.columns:
            self.datos["date_added"] = pd.to_datetime(
                self.datos["date_added"],
                errors="coerce"
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
                self.datos["duration"]
                .str.split(" ")
                .str[0]
            )
            self.datos["durationValue"] = pd.to_numeric(
                self.datos["durationValue"],
                errors="coerce"
            )
        return self.datos

    def limpiarDatos(self):
        self.eliminarDuplicados()
        self.reemplazarValoresNulos()
        self.convertirFecha()
        self.procesarDuracion()

        return self.datos
