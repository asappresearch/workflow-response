import fire
import csv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


path = "230616test_results/wf_prediction_epochs2/evaluation_tf.csv"
with open(path, "r") as fh:

    golds, preds=[], []
    for line in csv.DictReader(fh):
        gold = line["true_response"].strip()
        pred = line["response_1"].strip()

        golds.append(gold)
        preds.append(pred)

labels = list(set(golds))
label_dict = {v:i for i,v in enumerate(labels)}
print(label_dict)
# print(len(labels))
# exit()

cm = confusion_matrix(golds, preds, labels=labels)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
disp.plot()

plt.savefig("./fig.png")
