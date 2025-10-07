import cv2
import numpy as np

img = cv2.imread("images/my3.png")
img = cv2.resize(img, (int(img.shape[1]//2), int(img.shape[0]//2)))

img_color = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
edges = cv2.Canny(gray, 150, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
biggest = []  # naybilshi contury
for c in contours:
    if cv2.arcLength(c, True) > 1000:
        biggest.append(c)

for i in range(len(biggest)):
    for j in range(i + 1, len(biggest)):
        if cv2.boundingRect(biggest[i])[0] > cv2.boundingRect(biggest[j])[0]:
            x = biggest[i]
            biggest[i] = biggest[j]
            biggest[j] = x

x, y, w, h = cv2.boundingRect(biggest[0])
cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(img_color, "Roman Plyska", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(biggest[1])
cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(img_color, "Vlad Goviadovskiy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Original", img)
cv2.imshow("Canny", edges)
cv2.imshow("Squares", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
