import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread('images/giraffe.jpg')
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
img = cv2.GaussianBlur(img, (5, 5), 0)
img_copy = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([2, 0, 23])
upper = np.array([30, 198, 241])

cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min", "Trackbars", 2, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 30, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 198, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 23, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 241, 255, nothing)

last = {"H Min": lower[0], "H Max": upper[0], "S Min": lower[1], "S Max": upper[1], "V Min": lower[2], "V Max": upper[2]}

while True:
    pos = {k: cv2.getTrackbarPos(k, "Trackbars") for k in last}
    for k in pos:
        if pos[k] != last[k]:
            last[k] = pos[k]

    lower = np.array([last["H Min"], last["S Min"], last["V Min"]])
    upper = np.array([last["H Max"], last["S Max"], last["V Max"]])

    mask = cv2.inRange(hsv, lower, upper)
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    img_draw = img_copy.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            per = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                ar = round(w / h, 2)
                comp = round((4*np.pi * area)//(per)**2, 2)
                approx = cv2.approxPolyDP(cnt, 0.01*per, True)
                if len(approx) == 3:
                    shape = "Triangle"
                elif len(approx) == 4:
                    shape = "Quadratic"
                elif len(approx) > 8:
                    shape = "Circle"
                else:
                    shape = "lowtab"
                cv2.drawContours(img_draw, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(img_draw, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(img_draw, f"shape:{shape}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_draw, f"A:{int(area)}, p:{int(per)}", (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_draw, f"AR:{ar}, C:{comp}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("orih", img_draw)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
