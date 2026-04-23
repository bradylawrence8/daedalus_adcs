import spidev
import gpiozero
import pigpio as pg
import time
import math
import socket
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C

NUM_SENSORS = 3

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000
spi.mode = 0b01

i2c0 = busio.I2C(board.D1, board.D0) 

bno = BNO08X_I2C(i2c0)

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)


def read_angles():
    tx = []
    for _ in range(NUM_SENSORS):
        tx.extend([0xFF, 0xFF])

    rx = spi.xfer2(tx)

    values = []

    for i in range(NUM_SENSORS):
        high = rx[2*i]
        low  = rx[2*i + 1]

        value = ((high << 8) | low) & 0x3FFF
        value = value/16384.0

        values.append(value)

    return values
read_angles()
try:
    starttime = [time.perf_counter(), time.perf_counter(), time.perf_counter()]
    curstate = [0, 0, 0]
    spikecount = [0, 0, 0]
    rpm = [0, 0, 0]
    time0 = time.perf_counter()
    tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsocket.bind( ("0.0.0.0", 8000) )
    tcpsocket.listen(2)
    (client, (ip,port) ) = tcpsocket.accept()
    while True:
        r = read_angles()
        curtime = time.perf_counter()
        for i in range(3):
            response = r[i]
            if response > 0.9 and curstate[i] == 0:
                curstate[i] = 1
                spikecount[i] +=1
            if response < 0.1 and curstate[i] == 1:
                curstate[i] = 0

            if spikecount[i] == 7:
                curtime = time.perf_counter()
                rpm[i] = 60/(curtime-starttime[i])
                starttime[i] = curtime
                spikecount[i] = 0
        accel_x, accel_y, accel_z = bno.acceleration
        gyro_x, gyro_y, gyro_z = bno.gyro
        accelstring = str(accel_x) + "," + str(accel_y) + "," + str(accel_z) + ","
        gyrostring = str(gyro_x) + "," + str(gyro_y) + "," + str(gyro_z) + ","
        rpmstring = str(rpm[0]) + "," + str(rpm[1]) + "," + str(rpm[2]) + ";"
        outputstring = accelstring + gyrostring + rpmstring
        client.send(bytes(outputstring, 'utf8'))
except KeyboardInterrupt:
    spi.close()
