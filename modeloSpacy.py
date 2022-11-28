import spacy
import csv
import random
import time
import numpy as np
import pandas as pd
import re
import string

from spacy.util import minibatch, compounding
import sys
from spacy import displacy
from itertools import chain

from sklearn.metrics import classification_report

def load_word_vectors(model_name, word_vectors):
    import spacy
    import subprocess
    import sys
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init-model",
                    "en",
                    model_name,
                    "--vectors-loc",
                    word_vectors
                        ]
                    )
    print (f"New spaCy model created with word vectors. File: {model_name}")


def Sort(sub_li):
  return(sorted(sub_li, key = lambda x: x[1],reverse=True))

def evaluate(tokenizer, textcat, test_texts, test_cats):
    docs = (tokenizer(text) for text in test_texts)
    preds = []
    for i, doc in enumerate(textcat.pipe(docs)):
        # print(doc.cats.items())
        scores = Sort(doc.cats.items())
        # print(scores)
        catList = []
        for score in scores:
            catList.append(score[0])
        preds.append(catList[0])

    # labels = ['AGAINST', 'FAVOR']
    labels = ['fake', 'legit']
    print(classification_report(test_cats, preds, labels=labels))

def trainSpacy(eleccion):
    # Leemos los distintos datasets
    fTrain = "corpus/train.csv"
    fDev = "corpus/dev.csv"
    fTest = "corpus/test.csv"

    print("---Read data---")

    # -----------------train file-------------------------
    train_data = pd.read_csv(fTrain)

    train_text = train_data.open_response.values
    train_category = train_data.Chapter.values

    # -----------------Dev file-------------------------
    dev_data = pd.read_csv(fDev)

    dev_text = dev_data.open_response.values
    dev_cat = dev_data.Chapter.values

    # -----------------test file-------------------------
    test_data = pd.read_csv(fTest)

    test_text = test_data.open_response.values
    test_category = test_data.Chapter.values

    print("---Complete reading data---")
    if eleccion==1:
        nlp = spacy.load('en_core_web_sm')      #pretrained wordvectors
    else:
        load_word_vectors("modelos/spacy/custome_embeddings","modelos/my_word_embeddings.txt")
        nlp = spacy.load('modelos/spacy/custome_embeddings')    #our custome wordvectors

    # Añadimos el pipe de textcat a nuestro nlp
    textcat = nlp.get_pipe("textcat")

    # Añadimos la etiqueta al text classifier
    textcat.add_label('1')
    textcat.add_label('2')
    textcat.add_label('5')
    textcat.add_label('11')
    textcat.add_label('12')
    textcat.add_label('13')
    textcat.add_label('16')
    textcat.add_label('18')
    textcat.add_label('20')
    textcat.add_label('22')
    textcat.add_label('23')

    # Descartamos las pipes que no nos interesan
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))

        for i in range(200):
            print('EPOCH: ' + str(i))
            start_time = time.clock()
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=50)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.3, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the test data
                evaluate(nlp.tokenizer, textcat, dev_text, dev_cat)
            print('Elapsed time' + str(time.clock() - start_time) + "seconds")
        with nlp.use_params(optimizer.averages):
            filepath = "modelos/spacy/modeloSpacy"
            nlp.to_disk(filepath)
    return nlp



