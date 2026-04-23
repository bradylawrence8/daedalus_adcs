from picamera2 import *
import time

def takepicture(CamNum):
	cam = Picamera2(CamNum)
	cam.start()
	cam.capture_file(f"Camera{CamNum}.jpg")
	cam.stop()
	cam.close()

takepicture(0)
takepicture(1)
takepicture(2)
takepicture(3)
