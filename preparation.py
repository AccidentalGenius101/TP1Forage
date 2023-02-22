import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/speeddating.csv")

# Exploration
print(df.head())
print(df.info())
print(df.describe())

# On enlève la première colonne qui ne contient rien
df = df.drop(columns=["has_null", "race", "race_o"])

# On convertit les colonnes binaire en format true/false
for column in df:
    i = 0
    if re.search("^d_", column):
        df = df.drop(columns=[str(column)])
    else:
        for x in df[str(column)]:
            if x == 'b\'0\'' or x == 'b\'female\'':
                x = False
            elif x == 'b\'1\'' or x == 'b\'male\'':
                x = True
            df.at[i, str(column)] = x
            i += 1

df_num = df.select_dtypes(include=['float64'])
df_
ss = StandardScaler()
df_scaled = pd.DataFrame(ss.fit_transform(df_num), columns = df.columns)

# Dataframe convertit en csv
df.to_csv("data/cleaned_speeddating.csv")
