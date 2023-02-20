import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data/speeddating.csv")

# Exploration
print(df.head())
print(df.info())
print(df.describe())


# On enlève les colonnes qui sont seulement une différence de deux autres colonnes
for column in df:
    if re.search("^d_", column):
        df = df.drop(columns=[str(column)])

# On enlève la première colonne qui ne contient rien
df = df.drop(columns=["has_null"])

# On enlève les colonnes ayant plus de 400 valeurs manquantes
missing_val = df.isnull().sum().sort_values(ascending=False)
for x in missing_val:
    if x >= 400:
        column = missing_val[missing_val == x].index[0]
        df = df.drop(columns=[str(column)])
print(df.isnull().sum().sort_values(ascending=False))

# On enlève les observations ayant des valeurs manquantes
df = df.dropna()

# Vérification des valeurs aberrantes
df_num = df.select_dtypes(exclude='object')
df_num.boxplot()
# plt.show()

# Comme la plupart des valeurs sont des valeurs de note entrée
# par les participants on considère qu'il n'y a pas de valeurs aberrantes

# Encode les variables catégorielles pour la pca
df_cat = df.select_dtypes(include='object')
df = df.drop(df_cat, axis=1)
enc = OneHotEncoder()
encoder_df = pd.DataFrame(enc.fit_transform(df_cat).toarray())
final_df = df.join(encoder_df)
final_df = final_df.dropna()

# Standardise les données pour utiliser la PCA
final_df = StandardScaler().fit_transform(final_df)

# PCA pour réduire le nombre de variables
pca = PCA(.80)
final_df = pd.DataFrame(pca.fit_transform(final_df))

# Dataframe convertit en csv
final_df.to_csv("data/cleaned_speeddating.csv")
