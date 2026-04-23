import numpy as np
import board
import time
import RPi.GPIO as gp
import busio
import cv2 as cv
import math
from adafruit_bno08x.i2c import BNO08X_I2C
#from picamera2 import Picamera2
import xlsxwriter
from EKFFunctions import *
from LQR import *
from sscv2 import *
from threading import Thread, Event

def camera_thread():
     while True:
          for i in range(3):
               takepicture(i)

def main_thread():
    print("Start Setup")
    #Define Pins (BCM Pins) (use command "pinout" in command line for pins)
    IMUSDA = board.D0
    IMUSCL = board.D1
    PWMPin1 = 19
    # PWMPin2 = XX
    # PWMPin3 = XX

    workbook = xlsxwriter.Workbook('dynamictestdata.xlsx')
    worksheet = workbook.add_worksheet()

    #Define Variables
    numOfStates = 7
    numOfMeasurements = 4
    dt = 0.1
    g = 9.81
    GyroNoise = 0.4
    biasNoise = 0.001
    ReactionWheelJ = np.array([[1],[1],[1]])
    Accel_inertial = np.array((0, 0, 1)).reshape(3,1)
    initalq = np.array((0,0,0,1)).reshape(4,1)
    initalgyrobias = np.array((0,0,0)).reshape(3,1)
    sun_inertial = np.array((0.869,0.408,0.280)).reshape(3,1)
    vf = np.array((Accel_inertial.flatten(),sun_inertial.flatten())).reshape(2,3).T

    # initialize cameras
    #cams = [Picamera2(0), Picameras2(1), Picameras2(2)]
    #, Picamera2(1), Picamera2(2), Picamera2(3)]
    #config = cams[0].create_still_configuration(main={"size": (1920, 1080)})

    # for some reason running each camera like once cuts the image capture time in three
    #for i in range(4):
        #cams[i].configure(config)
        #takepicture(cams[i])

    #setup Motor Drivers
    # Set up GPIO Pin
    # gp.setmode(gp.BCM)

    # Setup Pins as IN/OUT
    # gp.setup(PWMPin1,gp.OUT)
    # gp.setup(PWMPin2, gp.OUT)
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
        0.00075356, 0.0003311, 0.00012968,   # accel
        1.42108547e-14, 1.88737914e-14, 1.73472348e-16    # sun sensor
    ])
    R = R**2

    x = np.zeros((7,1))
    z = np.zeros((6,1))
    x[0:4] = initalq.reshape(4,1)
    x[4:7] = initalgyrobias.reshape(3,1)

    ang_vel_w = 0

    tic = time.perf_counter()
    #main loop
    print("Enter Loop")
    try:
        i = 0
        starttime = time.perf_counter()
        lastaccel = [0, 0, 0]
        last gyro = [0, 0, 0]
        while True:
            # DETERMINATION

            #Get IMU data
            currentAccel = readAccel(bno) #read from file
            currentAccel = np.asarray(currentAccel, dtype=float)
            worksheet.write(i, 5, currentAccel[0])
            worksheet.write(i, 6, currentAccel[1])
            worksheet.write(i, 7, currentAccel[2])
            curaccel = [currentAccel[0], currentAccel[1], currentAccel[2]]

            currentGyro = readGyro(bno)
            currentGyro = np.asarray(currentGyro, dtype=float)
            worksheet.write(i, 8, currentGyro[0])
            worksheet.write(i, 9, currentGyro[1])
            worksheet.write(i, 10, currentGyro[2])
            curgyro = [currentGyro[0], currentGyro[1], currentGyro[2]]

            if abs(curaccel[0]-lastaccel[0])<0.0001 and abs(curaccel[1]-lastaccel[1])<0.0001 and abs(curaccel[2]-lastaccel[2])<0.001:
                if abs(curgyro[0]-lastgyro[0])<0.0001 and abs(curgyro[1]-lastgyro[1])<0.0001 and abs(curgyro[2]-lastgyro[2])<0.0001:
                    bno.soft_reset()
            lastaccel = curaccel
            lastgyro = curgyro

            # Get sun sensor data
            currentSunVector = sunsensor()
            currentSunVector = np.asarray(currentSunVector, dtype=float)
            worksheet.write(i, 11, currentSunVector[0])
            worksheet.write(i, 12, currentSunVector[1])
            worksheet.write(i, 13, currentSunVector[2])

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
            xp = State7TransitionFcn(x,w,dt)
            xp[0:4] /= np.linalg.norm(xp[0:4])
            #F = F7_Jacobian(x,w,dt)
            F = np.zeros((7,7))
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
            worksheet.write(i, 0, time.perf_counter()-starttime)
            worksheet.write(i, 1, x[0])
            worksheet.write(i, 2, x[1])
            worksheet.write(i, 3, x[2])
            worksheet.write(i, 4, x[3])

            print(f"Quaterion: {x[0:4].flatten()}")
            i += 1






            #CONTROL ALG

            #"_b" is body, "_w" is reaction wheel

            # determinedq_b = x[0:4] #Kalman Filter Output
            # determinedw_b = w #Measured Gyro

            # measuredw_w = np.array((0,0,0)).reshape(3,1) #Measured Wheels speed

            # targetq = np.array((0,0,0,1)).reshape(4,1) #Target q
            # targetw = np.array((0,0,0)).reshape(3,1) #Target angular velocity in body

            # qerr = qMultiply(q_inv(targetq),determinedq_b)
            # werr = determinedw_b - targetw #Check this

            # Torque = LQR(qerr,werr)
            # print(f"Torque: {Torque.flatten()}")

            # ang_acc_req_w = Torque/-ReactionWheelJ # T = -J*wdot

            # ang_vel_w = ang_vel_w + ang_acc_req_w*dt
            # ang_vel_err_w = measuredw_w - ang_vel_w

            # PWMin = PWMGain*ang_vel_err_w

            #PWM
            #p1 = gp.PWM(PWMPin1, PWMin[0])
            #p1.start()
            #p2 = gp.PWM(PWMPin2, PWMin[1])
            #p2.start()
            #p3 = gp.PWM(PWMPin3, PWMin[2])
            #p3.start()
            print(" ")
            toc = time.perf_counter()
            dt = toc-tic
            tic = toc

    except KeyboardInterrupt:
            print('\nEnding')
            gp.cleanup()
            workbook.close()
            import MotorShutdown.py
            #p1.stop()
            #p2.stop()
            #p3.stop()

t1 = Thread(target=main_thread, args=())
t2 = Thread(target=camera_thread, args=())
t1.start()
t2.start()
