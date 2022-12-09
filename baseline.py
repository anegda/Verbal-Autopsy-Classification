import sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import preproceso

def baseline():
    # Leemos los distintos datasets
    fTrain = "corpus/train.csv"
    fDev = "corpus/dev.csv"
    fTest = "corpus/test.csv"

    print("---Read data---")

    # -----------------train file-------------------------
    train_df = pd.read_csv(fTrain)
    preproceso.docEmbeddingsTrain(train_df)
    train_df = preproceso.docEmbeddingsTest(train_df)

    # -----------------test file-------------------------
    test_df = pd.read_csv(fTest)
    test_df = preproceso.docEmbeddingsTest(test_df)

    print("---Complete reading data---")

    YTrain = train_df[["Chapter"]]
    XTrain = train_df[["card2vec"]]
    XTrain = pd.DataFrame(XTrain.card2vec.tolist(), index= XTrain.index)

    YTest = test_df[["Chapter"]]
    XTest = test_df[["card2vec"]]
    XTest = pd.DataFrame(XTest.card2vec.tolist(), index= XTest.index)

    f = open("Resultados/Resultados Baseline.txt", "w")
    for i in range(10):
        #UTILIZAMOS COMO BASELINE EL GRADIENT BOOSTING
        model = GradientBoostingClassifier(n_estimators=10, random_state=(i*5)).fit(XTrain, YTrain)

        #EVALUAMOS CON TEST
        print(model.score(XTest, YTest))
        f.write(str(model.score(XTest, YTest))+"\n")
