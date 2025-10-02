import cv2
import numpy as np

img = np.zeros((400,600,3), np.uint8)
# img[:]= 3, 252, 40





cv2.rectangle(img,(0, 0), (600, 400),(200,200,180), thickness=cv2.FILLED)

image = cv2.imread("images/photo.png")

image = cv2.resize(image, (120, 180))

x, y = 30, 30
img[y:y+180, x:x+120] = image

cv2.line(img, (10, 10), (590, 10), (105, 0, 0), 2)
cv2.line(img, (590, 10), (590, 390), (105, 0, 0), 2)
cv2.line(img, (590, 390), (10, 390), (105, 0, 0), 2)
cv2.line(img, (10, 10), (10, 390), (105, 0, 0), 2)


cv2.putText(img, "Plyska Roman", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
cv2.putText(img, "Computer Vision Student", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
cv2.putText(img, "Email: rplyska@gmail.com", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)
cv2.putText(img, "Phone: +380990907802", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1)
cv2.putText(img, "11/05/2010", (200, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1)
cv2.putText(img, "OpenCV Business Card", (60, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)


image2 = cv2.imread("images/qr.png")

image2 = cv2.resize(image2, (120, 120))

x, y = 450, 250
img[y:y+120, x:x+120] = image2


cv2.imshow("Business Card", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()