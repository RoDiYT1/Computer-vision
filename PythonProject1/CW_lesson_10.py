import cv2
import numpy as np
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# function simple shape generation
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == ("triangle"):
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

X = []#spysok oznak
y = []#spysok mitok

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255)
}
shapes = ["circle", "square", "triangle"]

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3] #(b, g, r, alpha)
            features = list(mean_color)
            X.append(features)
            y.append(f'{color_name}_{shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
model = KNeighborsClassifier(n_neighbors=3)#neparne cyslo tochnist neyroankiye
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}%')

test_image = generate_image((0, 255, 0), "circle")
mean_color = cv2.mean(test_image)[:3]
prediction = model.predict([mean_color])
print(f'Prediction: {prediction[0]}')

cv2.imshow("img", test_image)