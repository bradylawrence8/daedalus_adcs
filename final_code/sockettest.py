import sys
import socket

tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

tcpsocket.connect(('172.30.185.61', 8000)) 

while True:
    data = tcpsocket.recv(1024)
    # decode to unicode string 
    strings = data.decode('utf8')
    #get the num
    num = float(strings)

    print(num)