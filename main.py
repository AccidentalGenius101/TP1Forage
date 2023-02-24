from river import stream
from river import tree
from river import metrics
from river import evaluate
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import tree as treeSK
import graphviz
from sklearn.model_selection import cross_validate

# -------Evaluating DT tree-------

# Data for DT
dfDT = pd.read_csv("data/cleaned_speeddatingDT.csv").drop(columns="Unnamed: 0")
XDT = dfDT.iloc[:, :-3]
yDT = dfDT.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(XDT, yDT, test_size=0.30, random_state=42)

# DT Model
start = time.time()
dt = treeSK.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)

# Visualising DT tree
dot_data = treeSK.export_graphviz(dt, out_file=None,
                                  feature_names=XDT.columns,
                                  filled=True, rounded=True,
                                  special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("DT")

# Evaluating model
scoring = ['accuracy', 'precision', 'f1', 'recall', 'roc_auc']
dt_results = cross_validate(dt, X_test, y_test, scoring=scoring)
print(dt_results)

end = time.time()
print("Time for DT: " + str(end - start))

# -------Evaluating incremental tree-------

# Data Stream for EFDT, VFDT and CVFDT
df = pd.read_csv("data/cleaned_speeddating.csv").drop(columns="Unnamed: 0")
X = df.iloc[:, :-3]
y = df.iloc[:, -1]
df = stream.iter_pandas(X, y)

# Différent modèles
efdt = tree.ExtremelyFastDecisionTreeClassifier()
vfdt = tree.HoeffdingTreeClassifier()
cvfdt = tree.HoeffdingAdaptiveTreeClassifier()
modelList = [efdt, vfdt, cvfdt]

# Différentes métriques
accuracy = metrics.Accuracy()
precision = metrics.Precision()
recall = metrics.Recall()
f1 = metrics.F1()
rocauc = metrics.ROCAUC()
metricsList = [accuracy, precision, recall, f1, rocauc]

for model in modelList:
    for metric in metricsList:
        evaluate.progressive_val_score(dataset=df, model=model, metric=metric, print_every=1000, show_time=True,
                                       show_memory=True)
        evaluate.reset()
        print("Model: " + str(model))
