import cv2
import numpy as np

img = np.zeros((400,400,3), np.uint8)
# img[:]= 3, 252, 40

# img[100:150, 200:280] = 3, 252, 40

cv2.rectangle(img,(100,100),(200,200),(0,255,0), thickness=cv2.FILLED)
cv2.line(img, (100, 100), (250, 250), (255, 255, 0), 8)
cv2.line(img, (img.shape[0]//4,img.shape[0]//2), ((img.shape[1]//4)*3,img.shape[1]//2), (255,255,255), 5)
cv2.line(img, (img.shape[0]//2,img.shape[0]//4), (img.shape[1]//2,(img.shape[1]//4)*3), (255,255,255), 5)

cv2.circle(img, (img.shape[0]//2, img.shape[1]//2), 100, (255,255,255), 5)
cv2.putText(img, "Komariv Ivan", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


print(img.shape)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()