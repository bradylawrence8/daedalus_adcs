import spidev
import time
import math

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000
spi.mode = 0b01

def read_encoder():
    # Send 0xFFFF (read command), then read response
    spi.xfer2([0xFF, 0xFF])

    # Read 16-bit response
    response = spi.xfer2([0x00, 0x00])

    # Combine bytes into 16-bit value
    angle = (response[0] << 8) | response[1]

    # Mask to 14 bits
    angle = angle & 0x3FFF

    return angle / 16384.0 #convert to 0-1

try:
    starttime = time.perf_counter()
    curstate = 0
    spikecount = 0
    rev = 0
    while True:
        response = read_encoder()
        #curtime = time.perf_counter()
        if response > 0.9 and curstate == 0:
            curstate = 1
            #dt = curtime-starttime
            #print("high", response, dt)
            #starttime = time.perf_counter()
            spikecount +=1
        if response < 0.1 and curstate == 1:
            curstate = 0
            #dt = curtime-starttime
            #print("low", response, dt)
            #starttime = time.perf_counter()

        if spikecount == 7:
            #rev += 1
            curtime = time.perf_counter()
            print(60/(curtime-starttime))
            starttime = curtime
            spikecount = 0
        #time.sleep(0.1)  

except KeyboardInterrupt:
    spi.close()
