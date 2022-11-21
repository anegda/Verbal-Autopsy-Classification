import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def limpiar_texto(texto):
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto

def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]

def lematizar(tokens):
    return [wnl.lemmatize(token) for token in tokens]

def eliminar_palabras_concretas(tokens):
    palabras_concretas = {"hospit", "die", "death", "doctor", "deceas", "person", "servic", "nurs", "client", "peopl", "patient",                   #ELEMENTOS DE HOSPITAL QUE NO APORTAN INFO SOBRE ENFERMEDAD
                          "brother", "father","respondetn","uncl","famili","member","husband","son", "daughter","marriag",
                          "day", "year", "month", "april", "date", "feb", "jan", "time", "place","later","hour",                                    #FECHAS QUE NO APORTAN INFO SOBRE ENFERMEDD
                          "interview", "opinion", "thousand", "particip", "admit", "document", "inform", "explain", "said", "respond","interviewe",                                                                                                #PALABRAS QUE TIENEN QUE VER CON LA ENTREVISTA
                          "write", "commend", "done", "told", "came", "done", "think", "went", "took", "got",                                       #OTROS VERBOS
                          "brought","becam","start",
                          "even", "also", "sudden", "would", "us", "thank","alreadi","rather","p","none","b",                                       #PALABRAS QUE NO APORTAN SIGNIFICADO
                          "caus", "due", "suffer", "felt", "consequ"}                                                                               #PALABRAS SEGUIDAS POR SINTOMAS


    return [token for token in tokens if token not in palabras_concretas]

def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]

def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
    plt.savefig('Imagenes/Matrices/'+title+'.png')

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
        print(docProbArray)
        howMany=len(docProbArray);
        print("How Many");
        print(howMany);
        for doc_index in top_doc_indices:
            print(documents[doc_index])

def topicosReview(cuerpo, indice_review):
    # Cargo el modelo lda
    file = open("./modelos/lda.sav", "rb")
    lda = pickle.load(file)
    file.close()

    bow_review = cuerpo[indice_review]
    topicos = [0] * lda.num_topics
    # Indices de los topicos mas significativos
    #dist_indices = [topico[0] for topico in lda[bow_review]]
    # Contribución de los topicos mas significativos
    # dist_contrib = [topico[1] for topico in lda[bow_review]]
    dt = lda.get_document_topics(bow_review)
    dicc = dict(dt)
    valores = [i for i in dicc.values()]
    llaves = [i for i in dicc.keys()]
    valorMax = max(valores)
    iMax = valores.index(valorMax)
    indice = llaves[iMax]
    topicos[indice] = valorMax
    """for t in dt:
        topicos[t[0]]=t[1]"""
    return topicos

def diseaseToChapter(disease):
    #NOS BASAMOS EN ICD-11 version 02/2022: https://icd.who.int/browse11/l-m/en
    dictDC = {"Other Non-communicable Diseases": 20,
              "Diarrhea/Dysentery": 1, "Other Infectious Diseases": 1, "AIDS": 1, "Sepsis": 1, "Meningitis": 1, "Meningitis/Sepsis": 1, "Malaria": 1, "Encephalitis": 1, "Measles":1, "Hemorrhagic fever":1, "TB": 1,
              "Leukemia/Lymphomas": 2, "Colorectal Cancer": 2, "Lung Cancer": 2, "Cervical Cancer": 2, "Breast Cancer": 2, "Stomach Cancer": 2, "Prostate Cancer": 2, "Esophageal Cancer": 2, "Other Cancers":2,
              "Diabetes": 5,
              "Epilepsy": 8,
              "Stroke": 11, "Acute Myocardial Infarction": 11, "Other Cardiovascular Diseases": 11,
              "Pneumonia": 12, "Asthma": 12, "COPD": 12,
              "Cirrhosis": 13, "Other Digestive Diseases": 13,
              "Renal Failure": 16,
              "Preterm Delivery": 18, "Stillbirth": 18, "Maternal": 18, "Birth asphyxia": 18, "Other Defined Causes of Child Deaths": 18,
              "Congenital malformation": 20,
              "Bite of Venomous Animal": 22, "Poisonings": 22,
              "Road Traffic": 23, "Falls": 23, "Homicide": 23, "Fires": 23, "Drowning": 23, "Suicide": 23, "Violent Death": 23, "Other Injuries": 23}

    return dictDC[disease]

def topicosTest(review, diccionario):
    dfOld = review
    df = review[["open_response"]]

    # 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
    df["Tokens"] = df.open_response.apply(limpiar_texto)

    # 2.- Tokenizamos
    tokenizer = ToktokTokenizer()
    df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

    # 3.- Eliminar stopwords y digitos
    df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

    # 4.- ESTEMIZAR / LEMATIZAR ???
    df["Tokens"] = df.Tokens.apply(estemizar)

    # 5.- ELIMINAMOS PALABRAS CONCRETAS QUE APARECEN MUCHO PERO NO APORTAN SIGNIFICADO
    df["Tokens"] = df.Tokens.apply(eliminar_palabras_concretas)

    diccionario.filter_extremes(no_below=0.1, no_above = 0.7)
    cuerpo = [diccionario.doc2bow(review) for review in df.Tokens]

    documents = df["open_response"]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tf_vectorizer.fit_transform(documents.values.astype(str))

    topicos = []
    for i in range(len(documents)):
        topicos.append(topicosReview(cuerpo, i))

    df["Topicos"] = topicos
    df["newid"] = dfOld["newid"]  # guardamos los ids

    return df

def topicosTrain(df, num_Topics):
    # ---> Parte 1: https://elmundodelosdatos.com/topic-modeling-gensim-fundamentos-preprocesamiento-textos/
    #ruta = str(input("Introduce el path relativo (EJ: ./datasets/nombre.csv) :"))
    dfOld = df      #guardamos aqui las columnas que no modificamos pero si necesitamos posteriormente
    df = df[["open_response"]]

    # 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
    df["Tokens"] = df.open_response.apply(limpiar_texto)

    # 2.- Tokenizamos
    tokenizer= ToktokTokenizer()
    df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

    # 3.- Eliminar stopwords y digitos
    df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

    # 4.- ESTEMIZAR / LEMATIZAR ???
    df["Tokens"] = df.Tokens.apply(estemizar)
    #print(df.Tokens[0][0:10])

    # 5.- ELIMINAMOS PALABRAS CONCRETAS QUE APARECEN MUCHO PERO NO APORTAN SIGNIFICADO
    df["Tokens"] = df.Tokens.apply(eliminar_palabras_concretas)

    # ---> Parte 2: https://elmundodelosdatos.com/topic-modeling-gensim-asignacion-topicos/
    # Cargamos en el diccionario la lista de palabras que tenemos de las reviews
    diccionario = Dictionary(df.Tokens)
    #print(f'Número de tokens: {len(diccionario)}') #mostrar el numero se palabras

    # Reducimos el diccionario filtrando las palabras mas raras o demasiado frecuentes
    # no_below = mantener tokens que se encuentran en el a menos 10% de los documentos
    # no_above = mantener tokens que se encuentran en no mas del 80% de los documentos
    diccionario.filter_extremes(no_below=0.10, no_above = 0.75)
    diccionario.save("modelos/dicc")
    #print(f'Número de tokens: {len(diccionario)}')

    # Creamos el corpus (por cada token en el df) QUE ES UN ARRAY BOW
    cuerpo = [diccionario.doc2bow(review) for review in df.Tokens]

    # BOW de una review
    # print(corpus[5])

    documents = df["open_response"]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    tf_vectorizer.fit_transform(documents.values.astype(str))

    lda = LdaModel(corpus=cuerpo, id2word=diccionario,
               num_topics=num_Topics, random_state=42,
               chunksize=1000, passes=10,
               alpha=0.2 , eta=0.9)

    # Guardo el modelo
    file = open("./modelos/lda.sav", "wb")
    pickle.dump(lda, file)
    file.close()
    topicos = []
    for i in range(len(documents)):
        topicos.append(topicosReview(cuerpo, i))
    df["Topicos"] = topicos

    df["newid"] = dfOld["newid"]    #guardamos los ids
    df["Chapter"] = dfOld["gs_text34"].apply(diseaseToChapter)  #guardamos los chapters

    return df, diccionario
