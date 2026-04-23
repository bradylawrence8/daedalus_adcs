from picamera2 import Picamera2
import time

cams = [Picamera2(0), Picamera2(1), Picamera2(2), Picamera2(3)]
config = cams[0].create_still_configuration(main={"size": (3280, 2464)})

def takepicture(i):
	tic = time.perf_counter()
	cams[i].start()
	image = cams[i].capture_image("main")
	image.save("test_image.jpg")
	cams[i].stop()
	toc = time.perf_counter()
	print(toc-tic)

for i in range(4):
	cams[i].configure(config)
	takepicture(i)

while(True):
	key = input("enter camera number: ")
	if key == '0':
		takepicture(0)
	if key == '1':
		takepicture(1)
	if key == '2':
		takepicture(2)
	if key == '3':
		takepicture(3)

