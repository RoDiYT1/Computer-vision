import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')
#print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])


X = df[["area", "perimeter", "corners"]]
Y = df['label_enc']

model = keras.Sequential([
    layers.Dense(8, activation= "relu", input_shape= (3,)),
    layers.Dense(8, activation= "relu"),
    layers.Dense(8, activation= "softmax")
])

model.compile(optimizer="adam",loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(X, Y, epochs = 300, verbose = 0)

plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history["accuracy"], label = 'accuracy')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend()
plt.show()

test = np.array([25,20,0])
pred = model.predict(test)

print(f'imovirrnist klasu: {pred}')
print(f'model vyznachila {encoder.inverse_transform([np.argmax(pred)])}')