import gensim.corpora
import pandas as pd
pd.options.mode.chained_assignment = None
import evaluacion
import preproceso
import pickle

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
        #Llamada al método
        main()

    elif int(eleccion) == 2:
        print("Ha elegido WordEmbeddings + Flair")
        # Llamada al método
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