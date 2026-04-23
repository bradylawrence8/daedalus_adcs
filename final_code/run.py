from threading import Thread, Event
from time import sleep
from time import time

event = Event()

def modify_variable(var):
    t_int = 0
    while True:
        for i in range(len(var)):
            t0 = time()
            var[i] = time()
            t_int += time()-t0
        if event.is_set():
            break
    print(t_int)

time0 = time()
my_var = [time()]
my_var2 = [time()]
t = Thread(target=modify_variable, args=(my_var, ))
t2 = Thread(target=modify_variable, args=(my_var2, ))
t.start()
t2.start()
while True:
    try:
        print(my_var[0]-time0, my_var2[0]-time0)
        sleep(1)
    except KeyboardInterrupt:
        event.set()
        break
t.join()
t2.join()
print(time()-time0)
print(my_var, my_var2)
