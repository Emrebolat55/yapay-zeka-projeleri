import pandas as pd 
import matplotlib.pyplot as plt     
from sklearn import linear_model

#veri setini impor ediyoruz
df = pd.read_csv("multilinearregression.csv",sep=";")
#veri setimiz görelim
print(df.head())

#liner regression modelini tanımlayalım
reg = linear_model.LinearRegression()
reg.fit(df[["alan","odasayisi","binayasi"]],df["fiyat"])

#prediction yapalım
print(reg.predict([[250,15,10]]))#örnek bir tahmin
print(reg.predict([[230,4,10],[230,6,0],[355,3,20]])) # çoklu tahmin de yapabiliriz [530243.29284619 586097.7833456  616657.45791365]

#katsayıları kontrol etmek için kullanılır
print(reg.coef_)
