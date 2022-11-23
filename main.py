import gensim.corpora
import pandas as pd
pd.options.mode.chained_assignment = None
import evaluacion
import preproceso
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import numpy as np

def apartadoComun():
    """Esto es lo mismo para todos los modelos.
    Hacemos el overSample, el split_train_test, etc."""
    f = "datasets/train.csv"
    df = pd.read_csv(f)
    # ESTAS COLUMNAS (en principio) NO SIRVEN PERO PODRÍAMOS USARLAS EN UN FUTURO PARA SEGMENTAR
    # Y VER COMO APRENDEN NUESTROS MODELOS EN DISTINTOS RANGOS DE EDADES, SEXOS, ETC.
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('module', axis=1)
    df = df.drop('age', axis=1)
    df = df.drop('sex', axis=1)
    df = df.drop('site', axis=1)
    df = preproceso.diseasesToChapters(df)
    df = df.drop('gs_text34', axis=1)

    print(df.head(5))       #IMPRIMIMOS 5
    # Esquema de validación método holdout
    # TODO: PROBAR bootstrapping
    dfTrain, dfTest = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['Chapter']])

    #No están igualmente distribuidas y hay pocas instancias => oversampling
    X_train = dfTrain.drop('Chapter', axis=1)
    Y_train = np.array(dfTrain['Chapter'])
    ros = RandomOverSampler(random_state=42) #TODO: MIRAR EL USO DE DICT
    X_train, Y_train = ros.fit_resample(X_train, Y_train)

    X_train['Chapter'] = Y_train
    print(X_train['Chapter'].value_counts())
    X_train.to_csv("Resultados/Prueba.csv")
    return df

def LDA_Flair():
    df = apartadoComun()

    df = preproceso.topicosTrain(df, 26, 0.2, 0.9)
    df.to_csv('Resultados/ResultadosPreprocesoLDA.csv')

    #Nos quedamos unicamente con las columnas que nos interesan
    return 0

def WE_Flair():
    df = apartadoComun()

    df = preproceso.embeddingsTrain(df)
    df.to_csv('Resultados/ResultadosPreprocesoEmbeddings.csv')

    return 0

def LDA_Bert():
    return 0

def WE_Bert():
    return 0

def main():
    print('''BIENVENIDO AL CLASIFICADOR DE VERBAL AUTOPSY
    
            Previamente hay que tener instaladas las siguientes librerías:
                - pandas
                - numpy
                - matplotlib
                - sklearn
                - seaborn
                - scikitplot
                - nltk
    
            Pulse el número según lo que que desee ejecutar:
                (1) LDA Topic Modeling + Flair
                (2) WordEmbeddings + Flair 
                (3) LDA Topic Modeling + Bert
                (4) WordEmbeddings + Bert 
                (5) Salir
    
            By Ane García\n''')

    eleccion = input()

    if int(eleccion) == 1:
        print("Ha elegido LDA Topic Modeling + Flair")
        LDA_Flair()
        main()

    elif int(eleccion) == 2:
        print("Ha elegido WordEmbeddings + Flair")
        WE_Flair()
        main()

    elif int(eleccion) == 3:
        print("Ha elegido LDA Topic Modeling + Bert")
        #Llamada al método
        main()

    elif int(eleccion) == 4:
        print("Ha elegido WordEmbeddings + Bert")
        # Llamada al método
        main()

    elif int(eleccion) == 5:
        print("SALIENDO...")
        return

    else:
        print("Seleccion incorrecta\n\n")
        main()


if __name__ == "__main__":
    main()