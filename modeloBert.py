import pandas as pd
import numpy as np
import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
import gensim.downloader as api
from bertopic import BERTopic

# BUSCAR EN INTERNET
# Tutorial: https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert

def trainBert():
    dfTrain = pd.read_csv('corpus/train.csv')
    dfDev = pd.read_csv('corpus/dev.csv')
    dfTest = pd.read_csv('corpus/test.csv')

    ft = api.load('my_word_embeddings')
    topic_model = BERTopic(embedding_model=ft)
