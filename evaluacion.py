import pickle

import numpy as np
# Calculating accuracy score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score
import scikitplot.metrics as skplt

def evaluar(referencias, clusters ,y_train):
    idx, cluster = list(zip(*clusters))
    cluster = np.asarray(cluster)
    labels = []
    for i in range(len(cluster)):
        labels.append(referencias[cluster[i]])

    print("La accuracy es:", accuracy_score(labels, y_train))
    print("La precision es:", precision_score(labels, y_train, average='weighted'))
    print("El f1 score es:", f1_score(labels, y_train, average='weighted'))

    error = {"aciertos":0, "errores":0}
    for y, label in zip(list(y_train), labels):
        if y-label == 0:
            error["aciertos"] = error["aciertos"]+1
        else:
            error["errores"] = error["errores"] + 1
    print(error)
    errorTotal = error["errores"]/(error["errores"]+error["aciertos"])
    print("El error es de: " + str(errorTotal))

    skplt.plot_confusion_matrix(labels, y_train)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig('Imagenes/matrizDeConfusion.png')
    plt.show()