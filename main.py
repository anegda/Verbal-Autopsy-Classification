import pandas as pd

import aleatorio

pd.options.mode.chained_assignment = None
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np


import preproceso
import modeloFlair
import modeloSpacy
import baseline
import aleatorio

def apartadoComun():
    """Esto es lo mismo para todos los modelos.
    Hacemos el overSample, el split_train_test, etc."""
    f = "datasets/train.csv"
    df = pd.read_csv(f)
    # ESTAS COLUMNAS (en principio) NO SIRVEN PERO PODRÍAMOS USARLAS EN UN FUTURO PARA SEGMENTAR
    # Y VER COMO APRENDEN NUESTROS MODELOS EN DISTINTOS RANGOS DE EDADES, SEXOS, LUGARES, ETC.
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('age', axis=1)
    df = df.drop('sex', axis=1)
    df = df.drop('site', axis=1)
    df = preproceso.diseasesToChapters(df)

    print(df.head(5))       #IMPRIMIMOS 5
    # Esquema de validación método holdout
    dfTrain, dfDev = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['Chapter']])

    # Separamos en atributos y clases
    X_train = dfTrain.drop('Chapter', axis=1)
    Y_train = np.array(dfTrain['Chapter'])

    # No están igualmente distribuidas y hay pocas instancias => oversampling
    distribucion = {1: 725, 2:245, 5:245, 11:573, 12:497, 13:245, 16:245, 18:819, 20:276, 22:245, 23:545}   #MINIMO LAS CLASES MINORITARIAS SON UN 30% DE LA CLASE MAYORITARIA
    ros = RandomOverSampler(random_state=42, sampling_strategy=distribucion)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    X_train['Chapter'] = Y_train
    X_train.to_csv("corpus/train.csv")

    # Guardamos el dev creado
    dfDev.to_csv("corpus/dev.csv")

    # Realizamos parecido con el test
    fTest = "datasets/test.csv"
    fLabelsRaw = "datasets/raw+labels.csv"
    dfTest = pd.read_csv(fTest)
    dfTestLabels = pd.read_csv(fLabelsRaw)

    dfTest = dfTest[['newid','module','open_response']]

    dfTestLabels = dfTestLabels[['newid', 'gs_text34', 'module']]
    dfTest = pd.merge(dfTest, dfTestLabels, how="left", on=["newid","module"])
    dfTest = preproceso.diseasesToChapters(dfTest)
    dfTest.to_csv("corpus/test.csv")

    return X_train
def WE_Flair(eleccion):
    df = apartadoComun()

    preproceso.wordEmbeddingsTrain(df)

    modeloFlair.trainFlair(eleccion)
    return 0

def WE_Spacy(eleccion):
    df = apartadoComun()

    preproceso.wordEmbeddingsTrain(df)

    modeloSpacy.trainSpacy(eleccion)
    return 0

def main():
    print('''BIENVENIDO AL CLASIFICADOR DE VERBAL AUTOPSY
       
            Pulse el número según lo que que desee ejecutar:
                (1) WordEmbeddings + Flair
                (2) Custome WordEmbeddings + Flair 
                (3) WordEmbeddings + Spacy
                (4) Custome WordEmbeddings + Spacy 
                (5) Baseline Gradient Boosting
                (6) Clasificador Aleatorio
                (7) Salir
    
            By Ane García\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Ha elegido WordEmbeddings + Flair")
        WE_Flair(1)
        main()

    elif int(eleccion) == 2:
        print("Ha elegido WordEmbeddings Custome + Flair")
        WE_Flair(2)
        main()

    elif int(eleccion) == 3:
        print("Ha elegido WordEmbeddings + Spacy")
        WE_Spacy(1)
        main()

    elif int(eleccion) == 4:
        print("Ha elegido WordEmbeddings Custome + Spacy")
        WE_Spacy(2)
        main()

    elif int(eleccion) == 5:
        baseline.baseline()
        return

    elif int(eleccion) == 6:
        aleatorio.clasificarAleatoriamente()
        return

    elif int(eleccion) == 7:
        print("SALIENDO...")
        return

    else:
        print("Seleccion incorrecta\n\n")
        main()


if __name__ == "__main__":
    main()