from river import stream
from river import compose
from river import preprocessing
from river import tree
from river import metrics
from river import evaluate
import pandas as pd

df = pd.read_csv("data/cleaned_speeddating.csv")
X = df.iloc[:, :-3]
y = df.iloc[:, -1]
df = stream.iter_pandas(X, y)

model = compose.Pipeline(
    # preprocessing.StandardScaler(),
    tree.ExtremelyFastDecisionTreeClassifier()
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset=df, model=model, metric=metric, print_every=200)




