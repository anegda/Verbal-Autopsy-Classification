from flair.data import Corpus
from flair.datasets import ClassificationCorpus, CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, FastTextEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
import gensim.downloader as api
from bertopic import BERTopic

def modeloFlair(df):
    #NOS QUEDAMOS UNICAMENTE CON LAS COLUMNAS QUE NECESITAMOS
    df = df[["clean_text"],["Chapter"]]

    #CREAMOS EL CLASSIFCATION CORPUS:  https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md#reading-a-text-classification-dataset
    data_folder = "/corpus"
    column_name_map = {6: "text", 8: "label_topic", 7: "label_subtopic"}
    corpus = CSVClassificationCorpus(data_folder, column_name_map, skip_header=True, delimiter=',')
    label_dict = corpus.make_label_dictionary(label_type="class")

    #CUSTOM WORD EMBEDDINGS
    ft = api.load('my_word_embeddings')
    #topic_model = BERTopic(embedding_model=ft)

    #LISTA DE WORDEMBEDDINGS
    word_embeddings = [
        WordEmbeddings('en-glove'),
        WordEmbeddings('en-crawl'),
        FlairEmbeddings('multi-X-fast'),
    ]

    #INICIALIZAMOS EL DOCUMENT EMBEDDINGS PASANDOLE UNA LISTA DE LOS WORD EMBEDDINGS
    document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                       hidden_size=512,
                                                                       reproject_words=True,
                                                                       reproject_words_dimension=256,
                                                                       rnn_type='LSTM',
                                                                       )

    #CREAMOS EL TEXT CLASSIFIER
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type="class")

    #INICIALIZAOS EL TEXT CLASSIFIER TRAINER
    trainer = ModelTrainer(classifier, corpus)

    #EMPEZAMOS EL ENTRENAMIENTO
    trainer.train('modelos/flair-VA', mini_batch_size=50, learning_rate=0.001, max_epochs=100)