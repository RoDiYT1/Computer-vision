import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures2.csv')

# Label encode
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# Features include new ratio feature
X = df[["area", "perimeter", "corners", "area_perimeter_ratio"]]
Y = df['label_enc']

# Updated architecture
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(len(df['label_enc'].unique()), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X, Y, epochs=400, verbose=0)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend()
plt.show()

test = np.array([[25, 20, 0, 25/20]])
pred = model.predict(test)

print(f'імовірність класів: {pred}')
print(f'модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')
