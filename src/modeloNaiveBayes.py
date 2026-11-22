from sklearn.naive_bayes import GaussianNB


class ModeloNaiveBayes:
    """
    Naive Bayes gaussiano: calcula, para cada clase (valoración alta o baja),
    la probabilidad de ver los valores de cada variable y combina todas
    asumiendo que las variables son independientes entre sí.
    No necesita escalar las variables: usa la media y la varianza de cada una.
    """

    def __init__(self):
        pass

    def estimador(self):
        return GaussianNB()
