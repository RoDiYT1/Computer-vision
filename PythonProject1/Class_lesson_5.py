import cv2
import numpy as np

img = cv2.imread('images/giraffe.jpg')
img = cv2.resize(img, (img.shape[1]*2, img.shape[0] * 2))
img = cv2.GaussianBlur(img, (5, 5), 0)


img_copy = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([2, 0, 23])
upper = np.array([30, 198, 241])
mask = cv2.inRange(hsv, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h, 2)# ДОпомагає відрізняти відношення сторін
            compactness = round((4*np.pi * area)// (perimeter)**2, 2) # Міра округлості обєкта
            approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Quadratic"
            elif len(approx) > 8:
                shape = "Circle"
            else:
                shape = "lowtab"
            cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0))
            cv2.putText(img_copy, f"shape:{shape}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img_copy, f"A:{int(area)}, p:{int(perimeter)}", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_copy, f"AR:{aspect_ratio}, C:{compactness}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)




cv2.imshow("orih", img_copy)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()