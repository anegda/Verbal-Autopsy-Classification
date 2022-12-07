import spacy
import csv
import random
import time
import numpy as np
import pandas as pd
import re
import string
import readline

from spacy.util import minibatch, compounding
import sys
from spacy import displacy
from itertools import chain

from sklearn.metrics import classification_report


def load_data_spacy(df):

    print(df['Chapter'].value_counts())

    texts = df['open_response'].tolist()
    cats = df['Chapter'].tolist()

    final_cats = []
    labels = [1, 2, 5, 8, 11, 12, 13, 16, 18, 20, 22, 23]
    for cat in cats:
        cat_list = {}
        for x in labels:
            if cat == x:
                cat_list[str(x)] = 1
            else:
                cat_list[str(x)]= 0
        final_cats.append(cat_list)

    train_data = list(zip(texts, [{"cats": cats} for cats in final_cats]))
    cats = [str(x) for x in cats]
    return train_data, texts, cats

def load_word_vectors(model_name, word_vectors):
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

    labels = ['1', '2', '5', '8','11', '12', '13', '16', '18', '20', '22', '23']
    print(test_cats)
    print(preds)
    print(classification_report(test_cats, preds, labels=labels))

def trainSpacy(eleccion):
    # Leemos los distintos datasets
    fTrain = "corpus/train.csv"
    fDev = "corpus/dev.csv"
    fTest = "corpus/test.csv"

    print("---Read data---")

    # -----------------train file-------------------------
    train_df = pd.read_csv(fTrain)
    train_data, train_text, train_cats = load_data_spacy(train_df)

    # -----------------Dev file-------------------------
    dev_df = pd.read_csv(fDev)
    dev_data, dev_text, dev_cats = load_data_spacy(dev_df)

    # -----------------test file-------------------------
    test_df = pd.read_csv(fTest)
    test_data, test_text, test_cats = load_data_spacy(test_df)

    print("---Complete reading data---")

    if eleccion==1:
        nlp = spacy.load('en_core_web_sm')      #pretrained wordvectors
    else:
        load_word_vectors("modelos/spacy/custome_embeddings","modelos/Embeddings/my_word_embeddings.txt")
        nlp = spacy.load('modelos/spacy/custome_embeddings')    #our custome wordvectors

    # Añadimos el pipe de textcat a nuestro nlp
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # Añadimos la etiqueta al text classifier
    textcat.add_label('1')
    textcat.add_label('2')
    textcat.add_label('5')
    textcat.add_label('8')
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

        for i in range(20):      #NUMERO DE EPOCHS
            print('EPOCH: ' + str(i))
            start_time = time.process_time()
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=50)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.3, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the test data
                evaluate(nlp.tokenizer, textcat, dev_text, dev_cats)
            print('Elapsed time' + str(time.process_time() - start_time) + "seconds")
        with nlp.use_params(optimizer.averages):
            filepath = "modelos/modeloSpacy"
            nlp.to_disk(filepath)

    print('---EVALUANDO TEST...---')
    evaluate(nlp.tokenizer, textcat, test_text, test_cats)
    return nlp



