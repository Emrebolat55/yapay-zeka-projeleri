import pandas as pd

df = pd.read_csv("imdb_top_1000.csv")
# ilk 5 satırı görüntülemek için
print(df.head(5))

# data setinin kopyasını oluşturma
df_kopya = df.copy()
print(df_kopya.head())

#data setindeki sutunları silmek için kullanılır (Colmn Drop)
df_yeni = df.drop(["Overview","Meta_score"],axis=1) #axis 1 ise column drop, 0 ise row drop eder. Defult=0
print(df_yeni.head())

#Satır silmek için (Row Drop)
df_yeni3 = df.drop([1])
print(df_yeni3.head())

#Datasetini Filtrelmek için
df_filtred = df[df["IMDB_Rating"]>=9] #ratingi 9 dan büyük olan filimler
print(df_filtred)

#Filtrelemek için 2. bir yöntem
df_filtred = df.query("IMDB_Rating>=9")
print(df_filtred)

#son 5 satırı görüntüleme için 
print(df.tail(5))

#data setini boyutunu görüntüler
print(df.shape)

#data setinin başlıklarına bakmak için 
print(df.columns)

#data set içindeki veri tipine bakalım
print(df.dtypes)

#data setinde eksik olan hücreleri görelim
print(df.isnull())

#data setindeki verlileri sıralamak
print(df.sort_values(by="imdb_score",ascending=False))

#yıllara göre flim sayılarnı bulmak için 
print(df["Releaset_Year"].value_counts())

#imdb raiting değeri 8in üstünde olan ve No_of_Votes 10000den çok olan filimleri göremek için
print(df.loc[(df["IMDB_Rating"]>=8) & (df["No_of_Votes"]>=100000)])

#dataframeye yeni bir satır eklemek için 
f = lambda x: (x["Runtime"].split())[0]
df["RuntimeYeni"] = df.apply(f,axis=1) 
# print(df.head())

#dataframedeki object değerini int'e çevirme
df["RuntimeYeni2"] = df["RuntimeYeni"].astype("int")
df_filtred = df.query("RuntimeYeni2 >=140")
# print(df_filtred)

# print(df.dtypes)#Data setinin özeliklerini görmek için

#şimdi runtimeyeniyi fazlalık olduğundan silelim
df = df.drop(["RuntimeYeni"],axis=1)
print(df)