import spidev
import time

NUM_SENSORS = 3

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000
spi.mode = 0b01


def read_angles():
    # Each sensor needs 2 bytes
    tx = []

    # Send angle read command (0xFFFF) to all sensors
    for _ in range(NUM_SENSORS):
        tx.extend([0xFF, 0xFF])

    # Transfer all at once
    rx = spi.xfer2(tx)

    # Parse returned data
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
    dt = [0, 0, 0]
    rpm = [0, 0, 0]
    s = [0, 0, 0]
    while True:
        response = read_angles()
        curtime = time.perf_counter()
        for i in range(NUM_SENSORS):
            if response[i] > 0.9 and curstate[i] == 0:
                curstate[i] = 1
                dt[i] += curtime-starttime[i]
                #print("high", response, dt)
                starttime[i] = time.perf_counter()
            if response[i] < 0.1 and curstate[i] == 1:
                s[i] += 1
                curstate[i] = 0
                dt[i] += curtime-starttime[i]
                #print("low", response, dt)
                starttime[i] = time.perf_counter()
            if s[i] == 1:
                rpm[i] = 60/dt[i]/7
                dt[i] = 0
                starttime[i] = time.perf_counter()
                s[i] = 0
        print("motor 1:", rpm[0], "motor 2:", rpm[1], "motor 3:", rpm[2])
        time.sleep(0.1)  
except KeyboardInterrupt:
    spi.close()
