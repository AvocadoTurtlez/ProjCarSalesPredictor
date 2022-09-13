import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('car_sales_dataset.txt', encoding='ISO-8859-1')
print(data)
sns.pairplot(data)
plt.show(block=True)

X = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Purchase_Amount'], axis = 1)
print(X)
print("X data Shape=",X.shape)

Y = data['Purchase_Amount']
print(Y)
Y = Y.values.reshape(-1,1)
print("Y Data Shape=",Y.shape)

scaler_in = MinMaxScaler()
X_scaled = scaler_in.fit_transform(X)
print(X_scaled)

scaler_out = MinMaxScaler()
Y_scaled = scaler_out.fit_transform(Y)
print(Y_scaled)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)
hl=25

model = Sequential()
model.add(Dense(hl, input_dim=5, activation='relu'))
model.add(Dense(hl, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())



model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['accuracy'])
epochs_hist = model.fit(X_train, Y_train, epochs=40,
                        batch_size=64, verbose=1,
                        validation_data=(X_test, Y_test))
print(epochs_hist.history.keys())

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

X_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
X_test_sample_scaled = scaler_in.transform(X_test_sample)
Y_predict_sample_scaled = model.predict(X_test_sample_scaled)
print('Predicted Y (Scaled) =', Y_predict_sample_scaled)
Y_predict_sample = scaler_out.inverse_transform(Y_predict_sample_scaled)
print('Predicted Y / Purchase Amount ', Y_predict_sample)