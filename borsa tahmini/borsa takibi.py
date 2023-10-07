import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Veri setini yükleme
data = pd.read_csv('stock_data.csv')

#'Kapanış' sütununu kullanarak verileri alın
dataset = data['Sentiment'].values
dataset = dataset.reshape(-1, 1)

# Verileri normalize etme
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Eğitim ve test veri setlerini oluşturma
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_data, test_data = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Verileri X ve y olarak ayırma
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 10
trainX, trainY = create_dataset(train_data, time_step)
testX, testY = create_dataset(test_data, time_step)

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Modeli eğitme
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=64, verbose=1)

# Tahmin yapma
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Normalleştirmeyi geri alma
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])

# Eğitim ve test veri setlerindeki ortalama kare hatalarını hesaplama
train_score = np.sqrt(np.mean(np.power(train_predict - trainY, 2)))
test_score = np.sqrt(np.mean(np.power(test_predict - testY, 2)))
print('Eğitim veri seti hata oranı:', train_score)
print('Test veri seti hata oranı:', test_score)

# Tahminleri görselleştirme
import matplotlib.pyplot as plt
# Eğitim verileri
plt.plot(scaler.inverse_transform(dataset))
plt.plot(np.concatenate([np.array([None] * (train_size + time_step)), train_predict[:, 0]]))
# Test verileri
plt.plot(np.concatenate([np.array([None] * (train_size + time_step)), test_predict[:, 0]]))
plt.show()
