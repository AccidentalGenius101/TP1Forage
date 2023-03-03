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

for key in dt_results:
    print("Model: DT " + str(key) + ": " + str(dt_results[str(key)].mean()*100))

end = time.time()
print("Time for DT: " + str(end - start) + " Node_count: " + str(dt.tree_.node_count) + " Max_depth: " + str(dt.tree_.max_depth)+ " n_leaves: " + str(dt.tree_.n_leaves))

# -------Evaluating incremental tree-------

# Data Stream for EFDT, VFDT and CVFDT
df = pd.read_csv("data/cleaned_speeddating.csv").drop(columns="Unnamed: 0")
X = df.iloc[:, :-3]
y = df.iloc[:, -1]
df = stream.iter_pandas(X, y)

# Différent modèles
modelList = [tree.ExtremelyFastDecisionTreeClassifier(), tree.HoeffdingTreeClassifier(),
             tree.HoeffdingAdaptiveTreeClassifier()]

# Différentes métriques
metricsList = [metrics.Accuracy(), metrics.Precision(), metrics.Recall(), metrics.F1(), metrics.ROCAUC()]

for metric in range(len(metricsList)):
    for model in range(len(modelList)):
        df = stream.iter_pandas(X, y)
        modelList = [tree.ExtremelyFastDecisionTreeClassifier(), tree.HoeffdingTreeClassifier(),
                     tree.HoeffdingAdaptiveTreeClassifier()]
        metricsList = [metrics.Accuracy(), metrics.Precision(), metrics.Recall(), metrics.F1(), metrics.ROCAUC()]
        evaluate.progressive_val_score(dataset=df, model=modelList[model], metric=metricsList[metric], print_every=8378,
                                       show_time=True, show_memory=True)
        print("Model: " + str(modelList[model]) + " " + str(metricsList[metric]) + ": ")
        print(modelList[model].summary)
