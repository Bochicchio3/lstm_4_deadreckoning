#! /usr/bin/env python

import socket

UDP_IP = '192.168.1.50'
UDP_PORT = 5555


def read_udp(str UDP_IP, int UDP_PORT):    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        data, addr = sock.recvfrom(10240)
        print("reived: ", data)
    

import numpy as np
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()