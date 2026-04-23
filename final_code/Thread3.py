import numpy as np
import time
import busio
from adafruit_bno08x.i2c import BNO08X_I2C

from EKFFunctions import *

def Thread3():
    #Define Variables
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

    # setup IMU
    i2c0 = busio.I2C(board.D1, board.D0)
    bno = BNO08X_I2C(i2c0)

    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_GYROSCOPE,
    )

    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_GYROSCOPE)

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

    #main loop
    while True:
        # DETERMINATION

        #Get IMU data
        currentAccel = readAccel() #read from file
        currentAccel = np.asarray(currentAccel, dtype=float)

        currentGyro = readGyro()
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
        
        print(x[0:4].flatten()) #Quaterion
        print(w) #gyro