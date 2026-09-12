import pandas as pd


class LectorDatos:

    def __init__(self, rutaArchivo):
        self.rutaArchivo = rutaArchivo

    def cargarDatos(self):
        datos = pd.read_csv(self.rutaArchivo)
        return datos
