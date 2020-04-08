#! /usr/bin/env python



import socket

UDP_IP = '192.168.1.50'
UDP_PORT = 5555

imu=[]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
while True:
    data, addr = sock.recvfrom(10240)
    imu.append(data)
    


