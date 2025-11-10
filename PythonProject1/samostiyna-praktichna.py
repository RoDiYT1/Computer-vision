import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = []
y = []

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}


for color_name, bgr in colors.items():
    for _ in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}%')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)[:3]
            color_label = model.predict([mean_color])[0]
            label = model.predict([mean_color])[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, label.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            perimeter = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)
            color = 0
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = round(w / h, 2)
                compactness = round((4 * np.pi * area) // perimeter ** 2, 2)
                approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
                corners = len(approx)

                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    if 0.9 <  aspect_ratio < 1.1:
                        shape = "Square"
                    else:
                        shape = "Rectangle"
                elif len(approx) > 12:
                    shape = "Circle"
                else:
                    shape = "Unknown"

                cv2.drawContours(frame, [cnt], -1, (0, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, f"{color_label} {shape} {area} {perimeter} {aspect_ratio} {compactness} {corners}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("img", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imshow("img", frame)
cap.release()
cv2.destroyAllWindows()