import board
import busio
import time
import xlsxwriter
from adafruit_bno08x.i2c import BNO08X_I2C

i2c0 = busio.I2C(board.D1, board.D0) 

bno = BNO08X_I2C(i2c0)

from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)

#workbook = xlsxwriter.Workbook('imu_data.xlsx')
#worksheet = workbook.add_worksheet()

i = 0
try:
	while True:
		try:
			accel_x, accel_y, accel_z = bno.acceleration
			gyro_x, gyro_y, gyro_z = bno.gyro
			print(accel_x, accel_y, accel_z)
			print(gyro_x, gyro_y, gyro_z)
			#worksheet.write(i, 0, accel_x)
			#worksheet.write(i, 1, accel_y)
			#worksheet.write(i, 2, accel_z)
			#worksheet.write(i, 3, gyro_x)
			#worksheet.write(i, 4, gyro_y)
			#worksheet.write(i, 5, gyro_z)
			time.sleep(0.2)
			i = i + 1
		except (RuntimeError, KeyError, ValueError, IndexError, OSError):
			continue
except KeyboardInterrupt:
    print("\nStopped.")

#workbook.close()
