import gpiozero
import pigpio as pg
import time

PWMPin = 26
PhasePin = 20

pi = pg.pi()

pi.set_mode(PWMPin, pg.OUTPUT)
pi.set_mode(PhasePin, pg.OUTPUT)

pi.write(PhasePin, 1)
pi.write(PWMPin, 0)

motor_speed = 100

pi.set_PWM_frequency(PWMPin, motor_speed)
pi.set_PWM_dutycycle(PWMPin, 0)
time.sleep(1)
while (True):
    duty = input("enter motor duty cycle: ")
    pi.set_PWM_dutycycle(PWMPin, duty)
    if duty == 0:
        break

pi.stop()


