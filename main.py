from river import stream
from river import tree
from river import metrics
from river import evaluate
import pandas as pd
import time
from sklearn import tree as treeSK

df = pd.read_csv("data/cleaned_speeddating.csv")
X = df.iloc[:, :-3]
y = df.iloc[:, -1]
df = stream.iter_pandas(X, y)

efdt = tree.ExtremelyFastDecisionTreeClassifier()
vfdt = tree.HoeffdingTreeClassifier()
cvfdt = tree.HoeffdingAdaptiveTreeClassifier()
dt = treeSK.DecisionTreeClassifier()



accuracy = metrics.Accuracy()
precision = metrics.Precision()
recall = metrics.Recall()
mcc = metrics.MCC()
f1 = metrics.F1()
rocauc = metrics.ROCAUC()
confusionmatrix = metrics.ConfusionMatrix()
crossentropy = metrics.CrossEntropy()
logloss = metrics.LogLoss()

metricsList = [accuracy, precision, recall, mcc, f1, rocauc, confusionmatrix, crossentropy, logloss]

dt = dt.fit(X, y)
treeSK.plot_tree(dt)

evaluate.progressive_val_score(dataset=df, model=efdt, metric=accuracy, print_every=200)
t



