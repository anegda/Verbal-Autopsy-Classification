from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
def clasificarAleatoriamente():
    fTest = "corpus/test.csv"
    test_df = pd.read_csv(fTest)
    YTest = test_df[["Chapter"]]

    clases = [1, 2, 5, 11, 12, 13, 16, 18, 20, 22, 23]
    f = open("Resultados/Resultados Aleatorio.txt", "w")
    for i in range(10):
        pred = np.random.choice(clases, 2233,p=[0.18, 0.05, 0.03, 0.14, 0.12, 0.03, 0.03, 0.2, 0.07, 0.02, 0.13])
        print(accuracy_score(YTest, pred))
        f.write(str(accuracy_score(YTest, pred)) + "\n")
