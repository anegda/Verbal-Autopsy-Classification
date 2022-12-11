import matplotlib.pyplot as plt
import scikitplot.metrics as skplt

f = open("modelos/flair-VA-Custome+Global/test.tsv")
gold = []
pred = []
for line in f:
    if line.__contains__("- Pred:"):
        pred.append(line.split()[2])
    if line.__contains__("- Gold:"):
        gold.append(line.split()[2])

skplt.plot_confusion_matrix(gold, pred)
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.savefig('Imagenes/matrizDeConfusion.png')
plt.show()