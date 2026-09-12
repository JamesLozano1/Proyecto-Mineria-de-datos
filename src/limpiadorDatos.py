import pandas as pd


class LimpiadorDatos:

    def __init__(self, datos):
        self.datos = datos

    def mostrarValoresNulos(self):
        return self.datos.isnull().sum()

    def eliminarDuplicados(self):
        self.datos = self.datos.drop_duplicates()
        return self.datos

    def convertirFecha(self):
        if "date_added" in self.datos.columns:
            self.datos["date_added"] = pd.to_datetime(
                self.datos["date_added"],
                errors="coerce"
            )

        return self.datos

    def limpiarDatos(self):
        self.eliminarDuplicados()
        self.reemplazarValoresNulos()
        self.convertirFecha()

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
