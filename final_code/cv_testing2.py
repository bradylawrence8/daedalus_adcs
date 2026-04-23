import cv2 as cv
import numpy as np
import math

# Load the image
image = cv.imread("capstone\Camera3.jpg")
dims = image.shape

# 1. Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 2. Apply Gaussian blur to reduce noise
# (11, 11) is the kernel size, 0 is the sigma value
blurred = cv.GaussianBlur(gray, (11, 11), 0)

ret, thr = cv.threshold(blurred, 240, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_img = np.zeros_like(image)
cv.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

for cnt in contours:
    area = cv.contourArea(cnt)
    # 1. Find the minimum enclosing circle for the current contour
    # This function returns the center coordinates (x, y) and the radius
    (x, y), radius = cv.minEnclosingCircle(cnt)

    # 2. Convert the center and radius to integer values, as cv2.circle expects int types
    center = (int(x), int(y))
    radius = int(radius)
    circArea = math.pi*radius**2
    truecenter = (center[0]-dims[1]/2, dims[0]/2-center[1])

    if (radius > 10):
        cv.circle(image, center, radius, (0, 255, 0), 2) # Draws a green circle with thickness 2
        print(truecenter)
        

# Display the result
cv.imshow("", image)
cv.waitKey(0)
cv.destroyAllWindows()
