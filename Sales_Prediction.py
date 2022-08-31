import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('car_sales_dataset.txt', encoding='ISO-8859-1')
print(data)
sns.pairplot(data)
plt.show(block=True)

inputs = data.drop(['Customer_Name', 'Customer_Email', 'Country', 'Gender', 'Age', 'Annual_Salary', 'Credit_Card_Debt', 'Net_Worth','Purchase_Amount'], axis = 1)
print(inputs)
print("Input data Shape=",inputs.shape)

output = data['Purchase_Amount']
print(output)
output = output.values.reshape(-1,1)
print("Output Data Shape=",output.shape)

scaler_in = MinMaxScaler()
input_scaled = scaler_in.fit_transform(inputs)
print(input_scaled)

scaler_out = MinMaxScaler()
output_scaled = scaler_out.fit_transform(output)
print(output_scaled)

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
epochs_hist = model.fit(input_scaled, output_scaled, epochs=20, batch_size=10, verbose=1, validation_split=0.2)
print(epochs_hist.history.keys())

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
input_test_sample_scaled = scaler_in.transform(input_test_sample)
output_predict_sample_scaled = model.predict(input_test_sample_scaled)
print('Predicted Output (Scaled) =', output_predict_sample_scaled)
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)