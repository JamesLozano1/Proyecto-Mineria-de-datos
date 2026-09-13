import matplotlib.pyplot as plt


class VisualizadorDatos:

    def __init__(self):
        pass

    def mostrarGraficaBarras(self, datos, titulo, etiquetaX, etiquetaY):

        datos.plot(kind="bar")

        plt.title(titulo)
        plt.xlabel(etiquetaX)
        plt.ylabel(etiquetaY)

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def mostrarGraficaLinea(self, datos, titulo, etiquetaX, etiquetaY):

        datos.plot(kind="line", marker="o")

        plt.title(titulo)
        plt.xlabel(etiquetaX)
        plt.ylabel(etiquetaY)

        plt.grid(True)

        plt.tight_layout()
        plt.show()
