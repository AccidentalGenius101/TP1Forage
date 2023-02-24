import pandas as pd
import re
import matplotlib.pyplot as plt

df = pd.read_csv("data/speeddating.csv")

# Exploration
print(df.head())
print(df.info())
print(df.describe())

# On enlève la première colonne qui ne contient rien
df = df.drop(columns=["has_null", "race", "race_o", "field"])

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

# On enlève les colonnes ayant plus de 400 valeurs manquantes
dfDT = df.copy()
missing_val = dfDT.isnull().sum().sort_values(ascending=False)
for x in missing_val:
    if x >= 400:
        column = missing_val[missing_val == x].index[0]
        dfDT = dfDT.drop(columns=[str(column)])
print(dfDT.isnull().sum().sort_values(ascending=False))

# On enlève les observations ayant des valeurs manquantes
dfDT = dfDT.dropna()

# Vérification des valeurs aberrantes
df_num = dfDT.select_dtypes(exclude='object')
df_num.boxplot()
# plt.show()

# Dataframe convertit en csv
# Ensemble de données pour les arbres incrémentals
df.to_csv("data/cleaned_speeddating.csv")
# Ensemble de données pour l'arbre normal
dfDT.to_csv("data/cleaned_speeddatingDT.csv")
