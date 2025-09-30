import cv2
import numpy as np

image = cv2.imread("images/homephoto.jpg")
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))


image2 = cv2.imread("images/hometext.jpg")


cv2.rectangle(image,(200,280),(400,600),(0,255,0), 2)
cv2.putText(image, "Roman Plyska", (223, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

print(image.shape)

cv2.imshow("me", image)
cv2.moveWindow("me", 100, 100)

cv2.waitKey(0)
cv2.destroyAllWindows()