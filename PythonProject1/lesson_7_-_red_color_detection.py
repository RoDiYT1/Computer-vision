import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])


lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

points = []
frcolor = (0, 255, 0)

while True:
    ret, frame = cap.read()


    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hcv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hcv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hcv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    cv2.line(frame, (0, 0), (0, 640), frcolor, 5)
    cv2.line(frame, (0, 480), (640, 480), frcolor, 5)
    cv2.line(frame, (640, 480), (640, 0), frcolor, 5)
    cv2.line(frame, (640, 0), (0, 0), frcolor, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        frcolor = (0, 0, 255)
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 150:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (0, 255, 0), 2)

            print(cx, cy)

            if cx != None:
                frcolor = (0, 255, 0)


    #             points.append((cx, cy))
    # for i in range(1, len(points)):
    #     if points[i - 1][1] is None or points[i] is None:
    #         continue
    #     cv2.line(frame, points[i - 1], points[i], (0, 0, 0), 2)


    cv2.imshow('Video', frame)
    cv2.imshow('Video2', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()