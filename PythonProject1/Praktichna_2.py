import cv2
import numpy as np

img = cv2.imread('images/obj.jpg')
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
img = cv2.GaussianBlur(img, (5, 5), 0)


img_copy = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([161, 21, 121])
upper_red = np.array([179, 225, 255])
lower_yellow = np.array([23, 62, 0])
upper_yellow = np.array([32, 255, 255])
lower_blue = np.array([99, 50, 0])
upper_blue = np.array([119, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_yellow)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in yellow_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: yellow", (x-30, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in red_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: red", (x-30, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in blue_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: blue", (x-30, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)




for cnt in contours:
    s = cv2.contourArea(cnt)
    if s > 200:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        color = 0
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = round(w / h, 2)
            compactness = round((4*np.pi * s)// perimeter ** 2, 2)
            approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Square"
            elif len(approx) > 12:
                shape = "Circle"
            else:
                shape = "Other"


            cv2.drawContours(img_copy, [cnt], -1, (0, 0, 0), 2)
            cv2.circle(img_copy, (cx, cy), 4, (0, 0, 0))
            cv2.putText(img_copy, f"shape:{shape}", (x-30, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(img_copy, f"A:{int(s)}, p:{int(perimeter)}", (x-30, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.putText(img_copy, f"AR:{aspect_ratio}, C:{compactness}", (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)




cv2.imshow("Shapes", img_copy)
cv2.imshow("mask", mask_total)
cv2.imshow("maskyll", mask_yellow)
cv2.imwrite("images/result.jpg", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()