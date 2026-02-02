import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

fig, ax = plt.subplots(1)
cam = cv.VideoCapture(0)

ret, frame = cam.read()

im = plt.imshow(frame, animated=True)


def updatefig(*args):
    ret, frame = cam.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 250, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            area = cv.contourArea(cnt)
            circarea = math.pi*radius**2
            circularity = area/circarea
            if circularity > 0.55:
                cv.circle(image, center, radius, (0, 255, 0), 2)
    im.set_array(image)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

cam.release()
cv.destroyAllWindows()