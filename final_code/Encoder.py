import spidev
import time
import xlsxwriter

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000
spi.mode = 0b01

workbook = xlsxwriter.Workbook('encoderdata.xlsx')
worksheet = workbook.add_worksheet()

def read_angle():
    # Send 0xFFFF (read command), then read response
    spi.xfer2([0xFF, 0xFF])

    # Read 16-bit response
    response = spi.xfer2([0x00, 0x00])

    # Combine bytes into 16-bit value
    angle = (response[0] << 8) | response[1]

    # Mask to 14 bits
    angle = angle & 0x3FFF

    return (angle * 360.0) / 16384.0 #convert to degrees

try:
    index = 0
    starttime = time.perf_counter()
    while True:
        angle_degrees = read_angle()
        curtime = time.perf_counter()

        print(f"Angle: {angle_degrees:.2f} degrees")
        worksheet.write(index, 0, curtime-starttime)
        worksheet.write(index, 1, angle_degrees)
        index += 1
        time.sleep(0.01)  

except KeyboardInterrupt:
    spi.close()
    workbook.close()
