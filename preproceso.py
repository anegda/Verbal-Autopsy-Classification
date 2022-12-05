import re

import pickle

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer

import gensim.corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

import texthero as hero
from texthero import preprocessing

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

# NECESARIOS PARA LA LIMPIEZA DE DATOS
nltk.download("stopwords", quiet=True)
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

def diseasesToChapters(df):
    df["Chapter"] = df["gs_text34"].apply(diseaseToChapter)  # guardamos los chapters
    return df

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

def docEmbeddingsTrain(df):
    #Tutorial: https://towardsdatascience.com/how-to-vectorize-text-in-dataframes-for-nlp-tasks-3-simple-techniques-82925a5600db

    #PROBAMOS CON OTRA LIMPIEZA DE DATOS
    custom_pipeline = [#preprocessing.fillna,    #se supone que no hay casillas vacías
                       preprocessing.lowercase,
                       preprocessing.remove_diacritics,
                       preprocessing.remove_punctuation,
                       preprocessing.remove_digits,
                       preprocessing.remove_stopwords,
                       preprocessing.remove_whitespace,
                       ]

    # Limpiamos el texto
    df['clean_text'] = hero.clean(df['open_response'], custom_pipeline)


    tokenizer = ToktokTokenizer()
    df["clean_text"] = df.clean_text.apply(tokenizer.tokenize)
    df["clean_text"] = df.clean_text.apply(estemizar)
    df["clean_text"] = df.clean_text.apply(eliminar_palabras_concretas)

    card_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(df.clean_text)]
    print(card_docs)

    # Inicializamos modelo
    model = Doc2Vec(vector_size=200, min_count=1, epochs=20)

    # Construimos vocabulario
    model.build_vocab(card_docs)

    # Entrenamos el modelo
    model.train(card_docs, total_examples=model.corpus_count, epochs=model.epochs)

    # Exportamos el modelo entrenado con pickle
    fname = "modelos/my_doc2vec_model"
    model.save(fname)

    # Generamos los vectores
    card2vec = [model.infer_vector((df['clean_text'][i])) for i in range(0, len(df['clean_text']))]

    # Añadimos al dataframe los vectores generados
    dtv = np.array(card2vec).tolist()
    df['card2vec'] = dtv

    return df

def wordEmbeddingsTrain(df):
    #Tutorial: https://towardsdatascience.com/how-to-vectorize-text-in-dataframes-for-nlp-tasks-3-simple-techniques-82925a5600db

    #PROBAMOS CON OTRA LIMPIEZA DE DATOS
    #PROBAMOS CON OTRA LIMPIEZA DE DATOS
    custom_pipeline = [#preprocessing.fillna,    #se supone que no hay casillas vacías
                       preprocessing.lowercase,
                       preprocessing.remove_diacritics,
                       preprocessing.remove_punctuation,
                       preprocessing.remove_digits,
                       preprocessing.remove_stopwords,
                       preprocessing.remove_whitespace,
                       ]

    # Limpiamos el texto
    df['clean_text'] = hero.clean(df['open_response'], custom_pipeline)


    tokenizer = ToktokTokenizer()
    df["clean_text"] = df.clean_text.apply(tokenizer.tokenize)
    df["clean_text"] = df.clean_text.apply(estemizar)
    df["clean_text"] = df.clean_text.apply(eliminar_palabras_concretas)

    card_docs = [row for row in df["clean_text"]]

    # Inicializamos modelo
    model = Word2Vec(vector_size=200, min_count=1, epochs=20)

    # Construimos vocabulario
    model.build_vocab(card_docs)

    # Entrenamos el modelo
    model.train(card_docs, total_examples=model.corpus_count, epochs=model.epochs)

    # GUARDAMOS DE LAS DOS MANERAS

    # PARA SPACY
    model.wv.save_word2vec_format("modelos/Embeddings/my_word_embeddings.txt")

    # PARA FLAIR
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('modelos/Embeddings/my_word_embeddings.txt', binary=False)
    word_vectors.save('modelos/Embeddings/word_embeddings_Flair')

    return df