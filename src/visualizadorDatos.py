import os
import matplotlib.pyplot as plt


class VisualizadorDatos:

    def __init__(self, mostrar=True, carpeta=None):
        self.mostrar = mostrar
        self.carpeta = carpeta
        self.contador = 0
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)

    def _finalizar(self, titulo):
        plt.tight_layout()
        if self.carpeta:
            self.contador += 1
            limpio = "".join(c if c.isalnum() else "_" for c in titulo.lower())[:50]
            plt.savefig(os.path.join(self.carpeta, f"{self.contador:02d}_{limpio}.png"), dpi=150)
        if self.mostrar:
            plt.show()
        else:
            plt.close()

    def mostrarGraficaBarras(self, datos, titulo, etiquetaX, etiquetaY):
        datos.plot(kind="bar")
        plt.title(titulo)
        plt.xlabel(etiquetaX)
        plt.ylabel(etiquetaY)
        plt.xticks(rotation=45)
        self._finalizar(titulo)

    def mostrarGraficaLinea(self, datos, titulo, etiquetaX, etiquetaY):
        datos.plot(kind="line", marker="o")
        plt.title(titulo)
        plt.xlabel(etiquetaX)
        plt.ylabel(etiquetaY)
        plt.grid(True)
        self._finalizar(titulo)
