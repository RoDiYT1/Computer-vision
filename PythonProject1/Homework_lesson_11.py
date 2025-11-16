import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_shape_data(n=50):
    data = []
    for _ in range(n):
        r = np.random.randint(20, 80)
        area = np.pi * r * r
        perimeter = 2 * np.pi * r
        aspect_ratio = 1.0
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        corners = 0
        apr = area / perimeter
        data.append([area, perimeter, corners, aspect_ratio, compactness, apr, "circle"])

    for _ in range(n):
        w = np.random.randint(30, 100)
        h = w
        area = w * h
        perimeter = 4 * w
        aspect_ratio = 1.0
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        corners = 4
        apr = area / perimeter
        data.append([area, perimeter, corners, aspect_ratio, compactness, apr, "square"])

    for _ in range(n):
        w = np.random.randint(30, 100)
        h = np.random.randint(20, 80)
        area = w * h
        perimeter = 2 * (w + h)
        aspect_ratio = w / h
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        corners = 4
        apr = area / perimeter
        data.append([area, perimeter, corners, aspect_ratio, compactness, apr, "rectangle"])

    for _ in range(n):
        b = np.random.randint(40, 100)
        h = np.random.randint(30, 70)
        area = 0.5 * b * h
        perimeter = b + h + np.sqrt(h**2 + (b/2)**2)
        aspect_ratio = b / h
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        corners = 3
        apr = area / perimeter
        data.append([area, perimeter, corners, aspect_ratio, compactness, apr, "triangle"])

    for _ in range(n):
        a = np.random.randint(20, 80)
        b = np.random.randint(30, 100)
        area = np.pi * a * b
        h = 3 * (a + b) - np.sqrt((3*a + b)*(a + 3*b))
        perimeter = np.pi * h
        aspect_ratio = a / b
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        corners = 0
        apr = area / perimeter
        data.append([area, perimeter, corners, aspect_ratio, compactness, apr, "ellipse"])

    return data

data = generate_shape_data(50)
df = pd.DataFrame(data, columns=["area","perimeter","corners","aspect_ratio","compactness","area_perimeter_ratio","label"])
df.to_csv("shapes.csv", index=False)

X = df.drop(columns=["label"])
y = pd.get_dummies(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(8, activation="relu"))
model.add(Dense(5, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=350, validation_data=(X_test, y_test))

plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over Epochs")
plt.savefig("accuracy_graph.png")
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.savefig("loss_graph.png")
plt.close()

model.save("shape_model.keras")

