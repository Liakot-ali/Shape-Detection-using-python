import cv2
import numpy as np
import matplotlib as plt

# Import Image
image = cv2.imread("assets/shape.jpg")

# Convert the color image into gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Gray",gray_image)

_, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

# cv2.imshow("Thresh", thresh_image)

# Find the contours of the image
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

# Detect the shape and write the shape name in the shape
for i, contour in enumerate(contours):
    if i == 0:
        continue
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(image, contour, 0, (0, 0, 255), 5)

    # Find the coordinates of a specific shape
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + 10)
    y_mid = int(y + h + 25)

    coords = (x_mid, y_mid)
    colour = (0, 0, 255)
    font = cv2.FONT_HERSHEY_DUPLEX

    # Define the shape
    if len(approx) == 3:
        cv2.putText(image, "Triangle", coords, font, 1, colour, 1)
    elif len(approx) == 4:
        cv2.putText(image, "Quadrilateral", coords, font, 1, colour, 1)
    elif len(approx) == 5:
        cv2.putText(image, "Pentagon", coords, font, 1, colour, 1)
    elif len(approx) == 6:
        cv2.putText(image, "Hexagon", coords, font, 1, colour, 1)
    else:
        cv2.putText(image, "Circle", coords, font, 1, colour, 1)

# Display the output image
cv2.imshow("Shapes Detected", image)
cv2.waitKey(0)