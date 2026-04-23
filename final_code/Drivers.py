import RPi.GPIO as gp
import time
PWMPin = 35
PhasePin = 36
#Etc

# Set up GPIO Pins
gp.setmode(gp.BOARD)

# Setup Pins as IN/OUT
gp.setup(PWMPin,gp.OUT)
gp.setup(PhasePin, gp.OUT)

# Usage Ex
gp.output(PhasePin, gp.HIGH)

#PWM Ex
rpm = 50
p = gp.PWM(PWMPin, 2000)
p.start(0) #in range 0-100
for i in range(0,rpm,1):
    p.ChangeDutyCycle(i) #Change from 0 to 50
    time.sleep(0.1)
time.sleep(1)
for i in range(rpm, -1, -1):
    p.ChangeDutyCycle(i) #Change from 50 to 0
    time.sleep(0.1)

p.stop()
gp.cleanup()
