import cv2 as cv
import numpy as np
import math
#from CameraPictures import *

def findsun(filepath):
    img = cv.imread(filepath)
    dims = img.shape
    #print(dims)

    # Perform operations on the frame here (e.g., convert to grayscale)
    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    C = (0, 0)
    rmax = 10
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > rmax:
            rmax = radius
            C = center
    cv.destroyAllWindows()
    if C == (0, 0):
        return C, dims
    else:
        #print(C)
        C = (C[0]-dims[1]/2, C[1]-dims[0]/2) # convert from top-left origin to center origin
        #print(C)
        return C, dims

def rotx(theta):
    return np.array([[1, 0, 0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])

def roty(theta):
    return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])

def rotz(theta):
    return np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

def plotAxis(m, ax, style):
    ax.plot([0, m[0, 0]], [0, m[1, 0]], [0, m[2, 0]], color='red', linestyle=style)
    ax.plot([0, m[0, 1]], [0, m[1, 1]], [0, m[2, 1]], color='blue', linestyle=style)
    ax.plot([0, m[0, 2]], [0, m[1, 2]], [0, m[2, 2]], color='green', linestyle=style)

def plotVec(v, ax, c, style):
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, linestyle=style)

def sunvec(c, dcm):
    v = np.array([c[0], c[1], 0])
    vn = v/np.linalg.norm(v)
    n = np.array([vn[1], -vn[0], vn[2]])
    return np.matmul(dcm, n)

def sunsensor():
    filepathlist = ["Camera0.jpg", "Camera1.jpg", "Camera2.jpg", "Camera3.jpg"]
    #camangles = [[0.9828, 1.0644, math.pi/2], [-0.9828, 2.0772, math.pi/2], [-2.1588, 1.0644, math.pi/2], [2.1588, 2.0772, math.pi/2]]
    camangle = [0.9828, 1.0644, 0]
    campos = [0.075, 0.05, 0.05]
    imagesizes = [[], [], [], []]
    usedcams = [0, 0]
    hfov = 155
    vfov = 115
    center1 = (0, 0)
    center2 = (0, 0)
    #takepicture(0)
    sv, dims = findsun("capstone\\Camera0.jpg")
    #print(sv)
    imagesizes[0] = dims
    #print(i, sv)
    center1 = sv
    if center1==(0, 0):
        #print("camera blackout")
        return (0, 0, 0) # camera blackout / eclipse
    elif center2==(0, 0):
        #print("one camera sees sun")
        d = 1.18737061111 # estimate "sun" distance from cameras
        hdist = d*math.sqrt(2*(1-math.cos(hfov/180*math.pi)))
        vdist = d*math.sqrt(2*(1-math.cos(vfov/180*math.pi)))
        sun = np.multiply(np.array(center1), np.array([hdist/imagesizes[usedcams[0]][0], vdist/imagesizes[usedcams[0]][1]]))
        #print([sun[0], sun[1], d])
        sun = [sun[0]+campos[0], sun[1]+campos[1], d+campos[2]]
        #print(sun)
        sun = sun/np.linalg.norm(sun)
        #print(sun)
        #dcm = np.matmul(rotz(camangle[2]), np.matmul(rotx(camangle[1]), rotz(camangle[0])))
        #print(dcm)
        dcm = np.array([[0.6969, -0.6536, -0.2952], [0.05512, 0.4592, -0.8866], [0.715, 0.6016, 0.3561]])
        #print(np.matmul(np.linalg.inv(dcm), sun))
        sunfinal = np.matmul(np.linalg.inv(dcm), sun)
        #sunfinal = np.matmul(rotx(-math.pi/4), sun)
        #sunfinal = np.matmul(roty(-math.pi/4), sunfinal)
        #sunfinal = np.matmul(rotx(math.pi/2), sunfinal)
        #print(center1)
        return sunfinal # one camera sees sun


sv = sunsensor()
print(sv)