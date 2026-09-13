import os
import pandas as pd


class ExportadorDatos:
    """
    Se encarga de generar los datasets finales del proyecto, ya con
    toda la información obtenida durante la limpieza y el análisis:

    - Dataset enriquecido: el dataset principal con las columnas
      calculadas durante el pipeline (durationValue, added_year,
      antiguedad_al_agregar), listo para usarse fuera del proyecto
      (dashboards, otros modelos, etc.).

    - Dataset resumen: reúne en un solo archivo los resultados de
      todos los análisis (conteos, promedios, distribuciones), sin
      importar que cada uno tenga una forma distinta (Series, describe(), etc.).
    """

    def __init__(self, rutaSalida="./CSV/resultados"):
        self.rutaSalida = rutaSalida
        os.makedirs(self.rutaSalida, exist_ok=True)

    def exportarDatosEnriquecidos(self, datos, nombreArchivo="netflix_dataset_procesado.csv"):
        rutaCompleta = os.path.join(self.rutaSalida, nombreArchivo)
        datos.to_csv(rutaCompleta, index=False)
        print(f"Dataset enriquecido exportado en: {rutaCompleta}")
        return rutaCompleta

    def _aFilas(self, seccion, resultado):
        """
        Convierte el resultado de un análisis (Series de value_counts,
        Series de describe(), dict, etc.) en filas (seccion, clave, valor)
        para poder combinar todos los análisis en una sola tabla, aunque
        cada uno tenga una estructura distinta.
        """
        filas = []

        if isinstance(resultado, pd.Series):
            for clave, valor in resultado.items():
                filas.append({"seccion": seccion, "clave": clave, "valor": valor})

        elif isinstance(resultado, dict):
            for clave, valor in resultado.items():
                filas.append({"seccion": seccion, "clave": clave, "valor": valor})

        elif isinstance(resultado, pd.DataFrame):
            for indice, fila in resultado.iterrows():
                for columna, valor in fila.items():
                    filas.append({
                        "seccion": seccion,
                        "clave": f"{indice}_{columna}",
                        "valor": valor
                    })

        else:
            filas.append({"seccion": seccion, "clave": "valor", "valor": resultado})

        return filas

    def exportarResumenAnalisis(self, resultados, nombreArchivo="resumen_analisis.csv"):
        """
        resultados: dict donde la clave es el nombre del análisis
        (ej. "Principales países") y el valor es el resultado tal como
        lo devuelve AnalizadorDatos (Series, describe(), dict, etc.)
        """
        todasLasFilas = []

        for seccion, resultado in resultados.items():
            todasLasFilas.extend(self._aFilas(seccion, resultado))

        resumen = pd.DataFrame(todasLasFilas)

        rutaCompleta = os.path.join(self.rutaSalida, nombreArchivo)
        resumen.to_csv(rutaCompleta, index=False)
        print(f"Resumen de análisis exportado en: {rutaCompleta}")
        return rutaCompleta
