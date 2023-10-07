import pandas as pd 
import matplotlib.pyplot as plt    
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("diabetes.csv")
print(data.head()) # ilk 5 seti göstermek için yapılır

seker_hastalığı = data[data.Outcome == 1]
saglıklı_ınsan = data[data.Outcome == 0]

#şimdi glikoza bakarak örnek bir çizim yapalım 
plt.scatter(saglıklı_ınsan.Age,saglıklı_ınsan.Glucose,color="green",label="sağlıklı",alpha = 0.4)
plt.scatter(seker_hastalığı.Age,seker_hastalığı.Glucose,color="red",label="diabet hastalığı",alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
print(plt.show())  

# x ve y eksenkerini belirliyelim
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)
# outcome sütunu(depented veriable)çıkarıp sadece indepented veribalsa bakıyoruz

#normalizasyon yapıyoruz x_ham_veri içerisindeki değrleri sadece 0 ve 1 arasında olacak şekilde hepisinini güncelliyoruz.
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

#önce 
print("Normalize öncesi ham veriler:\n")
print(x_ham_veri.head())

#sonra
print("\n\n\nNormalization sonrası yapay zekayaya vereceğimiz veriler")
print(x.head())

# train datamız ile test datamızı ayırıyoruz
# train datamız sağlıklı indan ile hasta insanı ayırt etmek için kullnalılacaktır.
#test datası ise test etmek için kullanılacaktır.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=1)

#knn modelini oluşturalım
knn = KNeighborsClassifier(n_neighbors= 3 ) #n_neighbor = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için test verilerimiz doğrulama değeri sonucu",knn.score(x_test,y_test))


# k kaç olmalı ?
# en iyi k değerini belirleyelim..
sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac, "  ", "Doğruluk oranı: %", knn_yeni.score(x_test,y_test)*100)
    sayac += 1

# Yeni bir hasta tahmini için:



