import time
from picamera2 import Picamera2
import numpy as np
from adafruit_bno08x.i2c import BNO08X_I2C
import busio
import board

def takepicture(cam):
	tic = time.perf_counter()
	cam.start()
	image = cam.capture_image("main")
	image.save("test_image.jpg")
	cam.stop()
	toc = time.perf_counter()
	print(toc-tic)
	
def readAccel(bno):
    ax, ay, az = bno.acceleration
    return np.array([[ax],[ay],[az]])

def readGyro(bno):
    gx, gy, gz = bno.gyro
    return np.array([[gx],[gy],[gz]])

IMUSDA = board.D0
IMUSCL = board.D1

cams = [Picamera2(0), Picamera2(1)]
config = cams[0].create_still_configuration(main={"size": (1920, 1080)})
for i in range(2):
	cams[i].configure(config)
	takepicture(cams[i])
      
i2c0 = busio.I2C(IMUSCL, IMUSDA)
bno = BNO08X_I2C(i2c0)

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)

tic = time.perf_counter()
takepicture(cams[0])
toc = time.perf_counter()
print("image capture in " + (toc-tic) + " seconds")
