import cv2
import numpy as np
import matplotlib.pyplot as plt

camera_frame = cv2.imread("/home/nitesh/cube_image.jpg")
# Convert the image to HSV color space
hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)

# Define the range for green color
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Create a mask for green color
mask = cv2.inRange(hsv, lower_green, upper_green)
print(cv2.countNonZero(mask))
# Bitwise-AND mask and original image
