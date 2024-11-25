import numpy as np  
import pandas as pd  
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")

düzeltme_mapping = {"Y":1 ,"N":0}

df["IseAlindi"] = df["IseAlindi"].map(düzeltme_mapping)
df["SuanCalisiyor?"] = df["SuanCalisiyor?"].map(düzeltme_mapping)
df["Top10 Universite?"] = df["Top10 Universite?"].map(düzeltme_mapping)
df["StajBizdeYaptimi?"] = df["StajBizdeYaptimi?"].map(düzeltme_mapping)
düzeltme_mapping_eğitim = {"BS":0 , "PS":1 ,"PHD":2 }

df["Egitim Seviyesi"] = df["Egitim Seviyesi"].map(düzeltme_mapping_eğitim)

y = df[["IseAlindi"]]
x = df.drop(["IseAlindi"], axis=1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

print(clf.predict([[1, 0, 1, 0, 0, 1]]))
