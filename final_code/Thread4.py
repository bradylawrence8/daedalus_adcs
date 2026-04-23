import numpy as np
import time
import pigpio as pg
import spidev

from EKFFunctions import *
from LQR import *

# NEED TO ADD
# Encoder functions 
# Encoder to rpm code  
def Thread4():
    #Define Pins (all pins are using GPIO pins numbering)
    # Drivers
    PWMPin1 = 5
    PWMPin2 = 13
    PWMPin3 = 26
    PHPin1 = 6
    PHPin2 = 16
    PHPin3 = 20
    PWMPIN_arr = [PWMPin1, PWMPin2, PWMPin3]
    PHPIN_arr = [PHPin1, PHPin2, PHPin3]

    # Encoders
    MOSI = 10
    MISO = 9
    CLK = 11
    SDA = 8

    #Define Variables
    ReactionWheelJ = []
    dt = []
    NUM_OF_MOTORS = 3
    #From EKF
    x = []
    w = []

    #setup Motor Drivers
    # Set up GPIO Pin
    pi = pg.pi()

    # Setup Pins as IN/OUT
    for i in range(NUM_OF_MOTORS):
        pi.set_mode(PWMPIN_arr[i], pg.OUT)
        pi.set_mode(PHPIN_arr[i], pg.OUT)

    #initalize Pins
    for i in range(NUM_OF_MOTORS):
        pi.write(PWMPIN_arr[i],  0)
        pi.write(PHPIN_arr[i], 1)

    # set up Encoders
    spi = spidev.SpiDev()
    spi.open(0,0)
    spi.max_speed_hz = 1000000
    spi.mode = 0b01

    read_angles()
    ang_vel_req_w = 0
    prevw_w = 0
    #main loop
    try:
        while True:
            #CONTROL ALG

            #"_b" is body, "_w" is reaction wheel
            determinedq_b = x[0:4] #Kalman Filter Output
            determinedw_b = w #Measured Gyro

            measuredw_w = np.array((0,0,0)).reshape(3,1) #Measured Wheels angular velocity

            targetq_b = np.array((0,0,0,1)).reshape(4,1) #Target q
            targetw_b = np.array((0,0,0)).reshape(3,1) #Target angular velocity in body

            qerr = qMultiply(q_inv(targetq_b),determinedq_b)
            werr = determinedw_b - targetw_b #Check this

            Torque = LQR(qerr,werr)
            print(Torque)

            ang_acc_req_w = Torque/-ReactionWheelJ # T = -J*wdot

            # measureda_w = (measuredw_w - prevw_w)/dt
            # prevw_w = measuredw_w
            # ang_acc_err_w = measureda_w - ang_acc_req_w

            ang_vel_w = ang_vel_w + ang_acc_req_w*dt
            ang_vel_err_w = measuredw_w - ang_vel_w

            fuckingGain = 0.5
            PWMin = fuckingGain*ang_vel_err_w
            print(PWMin)

            # PWM
            for i in range(NUM_OF_MOTORS):
                pi.set_PWM_frequency(PWMPIN_arr[i], PWMin[i]) 
            
            print(" ")

    except KeyboardInterrupt:
        print('\nEnding')
        spi.close()
        for i in range(NUM_OF_MOTORS):
            pi.set_PWM_dutycycle(PWMPIN_arr[i], 0)
        pi.stop()