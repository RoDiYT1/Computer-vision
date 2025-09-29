import cv2
image = cv2.imread('1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)
cv2.imshow("Gray", gray)

size = cv2.resize(image, (400, 300))
sizegray = cv2.resize(gray, (500, 700))
cv2.imshow("Size", size)
cv2.imshow("SizeGray", sizegray)

blurred = cv2.GaussianBlur(size, (7, 7), 0)
blurredgray = cv2.GaussianBlur(sizegray, (5, 5), 0)
cv2.imshow("Blurred", blurred)
cv2.imshow("Blurredgray", blurredgray)

border = cv2.Canny(blurred, 30, 100)
cv2.imshow("Canny", border)

cv2.waitKey(0)
cv2.destroyAllWindows()