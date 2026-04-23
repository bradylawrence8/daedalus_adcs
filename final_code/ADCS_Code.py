import numpy as np
import board
import time
import RPi.GPIO as gp
import busio
import cv2 as cv
import math
from adafruit_bno08x.i2c import BNO08X_I2C
from picamera2 import Picamera2

from EKFFunctions import *
from LQR import *

print("Start Setup")
#Define Pins (BCM Pins) (use command "pinout" in command line for pins)
IMUSDA = board.D0
IMUSCL = board.D1
PWMPin1 = 19
# PWMPin2 = XX
# PWMPin3 = XX

#Define Variables
PWMGain = 0.5
numOfStates = 7
numOfMeasurements = 4
Gyrodt = 0.1
Acceldt = 0.1
SunSensordt = 0.1
g = 9.81
GyroNoise = 0.4
biasNoise = 0.001
ReactionWheelJ = np.array([[1],[1],[1]])
Accel_inertial = np.array((0, 0, 1)).reshape(3,1)
initalq = np.array((0,0,0,1)).reshape(4,1)
initalgyrobias = np.array((0,0,0)).reshape(3,1)
sun_inertial = np.array((0,0,1)).reshape(3,1)
vf = np.array((Accel_inertial.flatten(),sun_inertial.flatten())).reshape(2,3).T

# initialize cameras
cams = [Picamera2(0), Picamera2(1), Picamera2(2), Picamera2(3)]
config = cams[0].create_still_configuration(main={"size": (1920, 1080)})

def takepicture(i, filepath):
	#tic = time.perf_counter()
	cams[i].start()
	image = cams[i].capture_image("main")
	image.save(filepath)
	cams[i].stop()
	#toc = time.perf_counter()
	#print(toc-tic)

# for some reason running each camera like once cuts the image capture time in three
filepathlist = ["Camera0.jpg", "Camera1.jpg", "Camera2.jpg", "Camera3.jpg"]
for i in range(4):
	cams[i].configure(config)
	takepicture(i, filepathlist[i])

#setup Motor Drivers
# Set up GPIO Pin
# gp.setmode(gp.BCM)

# Setup Pins as IN/OUT
gp.setup(PWMPin1,gp.OUT)
#gp.setup(PWMPin2, gp.OUT)
#print("Pins setup finished")

# setup IMU
i2c0 = busio.I2C(IMUSCL, IMUSDA)
bno = BNO08X_I2C(i2c0)

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)

#print("IMU setup finished")

#initalize Kalman filter matrixes and state vector. State 1-4 are quaterion, States 5-7 are gyro bias. Measurement 1-3 are accel, 4-6 are sun sensor
P = np.diag([1e-3,1e-3,1e-3,1e-3,  1e-4,1e-4,1e-4]) #uncertainity in the initial state
Q = np.zeros((7,7))
Q[0:4,0:4] = 1e-4*np.eye(4)
Q[4:7,4:7] = 1e-6*np.eye(3)
R = np.diag([
    1.61069607e-3, 1.73464517e-3, 6.66772820e-6,   # accel
    0.00169965, 0.00238374, 0.00030031    # sun sensor
])
R = R**2

x = np.zeros((7,1))
z = np.zeros((6,1))
x[0:4] = initalq.reshape(4,1)
x[4:7] = initalgyrobias.reshape(3,1)

ang_vel_w = 0

# sun sensor files
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
    camangles = [[0.9828, 1.0644, math.pi/2], [-0.9828, 2.0772, math.pi/2], [-2.1588, 1.0644, math.pi/2], [2.1588, 2.0772, math.pi/2]]
    imagesizes = [[], [], [], []]
    usedcams = [0, 0]
    hfov = 155
    vfov = 115
    center1 = (0, 0)
    center2 = (0, 0)
    for i in range(4):
        takepicture(i, filepathlist[i])
        sv, dims = findsun(filepathlist[i])
        imagesizes[i] = dims
        #print(i, sv)
        if sv!=(0, 0):
            if center1==(0,0):
                center1=sv
                usedcams[0] = i
            else:
                center2=sv
                usedcams[1] = i
    if center1==(0, 0):
        #print("camera blackout")
        return (0, 0, 0) # camera blackout / eclipse
    elif center2==(0, 0):
        #print("one camera sees sun")
        d = 48 # estimate "sun" distance from cameras
        hdist = d*math.sqrt(2*(1-math.cos(hfov/180*math.pi)))
        vdist = d*math.sqrt(2*(1-math.cos(vfov/180*math.pi)))
        sun = np.multiply(np.array(center1), np.array([hdist/imagesizes[usedcams[0]][0], vdist/imagesizes[usedcams[0]][1]]))
        sunfinal = [sun[0], sun[1], d]/np.linalg.norm([sun[0], sun[1], d])
        #print(center1)
        return sunfinal # one camera sees sun
    else:
        #print("two cameras see sun")
        dcm1 = np.matmul(rotz(camangles[usedcams[0]][2]), np.matmul(rotx(camangles[usedcams[0]][1]), rotz(camangles[usedcams[0]][0])))
        dcm2 = np.matmul(rotz(camangles[usedcams[0]][2]), np.matmul(rotx(camangles[usedcams[1]][1]), rotz(camangles[usedcams[1]][0])))
        sun1 = sunvec(center1, dcm1)
        sun2 = sunvec(center2, dcm2)
        sunfinal = np.cross(sun1, sun2)/np.linalg.norm(np.cross(sun1, sun2))
        #print(center1, center2)
        return sunfinal # two cameras see sun



#main loop
print("Enter Loop")
try:
    while True:
        # DETERMINATION
        tic = time.perf_counter()
        #Get IMU data
        currentAccel = readAccel(bno) #read from file
        currentAccel = np.asarray(currentAccel, dtype=float)

        currentGyro = readGyro(bno)
        currentGyro = np.asarray(currentGyro, dtype=float)

        # Get sun sensor data
        currentSunVector = sunsensor()
        currentSunVector = np.asarray(currentSunVector, dtype=float)

        #Make all measurements unit vectors
        currentAccel /= np.linalg.norm(currentAccel)
        if np.linalg.norm(currentSunVector) > 1e-8:
            currentSunVector /= np.linalg.norm(currentSunVector)
        else:
            currentSunVector = np.zeros(3)

        #define w
        wx = currentGyro[0]
        wy = currentGyro[1]
        wz = currentGyro[2]
        w = np.array((wx,wy,wz)).reshape(3,1)-x[4:7]

        #Kalman Filter
        #Predict
        xp = State7TransitionFcn(x,w,Gyrodt)
        xp[0:4] /= np.linalg.norm(xp[0:4])
        F = F7_Jacobian(x,w,Gyrodt)
        Pp = F@P@F.T + Q

        #Update
        z = np.vstack((currentAccel.reshape(3,1), currentSunVector.reshape(3,1)))
        zp = Measurement7Model(xp, sun_inertial)
        error = z-zp
        H = H7_Jacobian(xp, sun_inertial)

        # Correction
        S = H@Pp@H.T+R
        K = Pp@H.T@np.linalg.inv(S)

        x = xp+K@error
        P = (np.identity(numOfStates)-K@H)@Pp
        x[0:4] = x[0:4]/np.linalg.norm(x[0:4])

        print(f"Quaterion: {x[0:4].flatten()}")

        toc = time.perf_counter()
        print(toc-tic)




        #CONTROL ALG

        #"_b" is body, "_w" is reaction wheel

        determinedq_b = x[0:4] #Kalman Filter Output
        determinedw_b = w #Measured Gyro

        measuredw_w = np.array((0,0,0)).reshape(3,1) #Measured Wheels speed

        targetq = np.array((0,0,0,1)).reshape(4,1) #Target q
        targetw = np.array((0,0,0)).reshape(3,1) #Target angular velocity in body

        qerr = qMultiply(q_inv(targetq),determinedq_b)
        werr = determinedw_b - targetw #Check this

        Torque = LQR(qerr,werr)
        print(f"Torque: {Torque.flatten()}")

        ang_acc_req_w = Torque/-ReactionWheelJ # T = -J*wdot

        ang_vel_w = ang_vel_w + ang_acc_req_w*Gyrodt
        ang_vel_err_w = measuredw_w - ang_vel_w

        PWMin = PWMGain*ang_vel_err_w

        #PWM
        #p1 = gp.PWM(PWMPin1, PWMin[0])
        #p1.start()
        #p2 = gp.PWM(PWMPin2, PWMin[1])
        #p2.start()
        #p3 = gp.PWM(PWMPin3, PWMin[2])
        #p3.start()
        print(" ")

except KeyboardInterrupt:
        print('\nEnding')
        gp.cleanup()
        #p1.stop()
        #p2.stop()
        #p3.stop()
