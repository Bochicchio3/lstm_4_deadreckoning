# %%
#forse sta installando cose 
#  ah devi installare anche live share extension pack
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import os 

import tensorflow as tf

from tensorflow.keras import layers

# %%

# READ THE DATA


os.chdir('./home/alfredo/Desktop/PROJECTS/indoor_mapping/Oxford Inertial Odometry Dataset_2.0(1)/Oxford Inertial Odometry Dataset/handheld/data1/syn/')

imu1 = np.genfromtxt('imu1.csv', delimiter=',')
vi1 = np.genfromtxt('vi1.csv', delimiter=',')

gyro1=imu1[:,4:7]
acc1=imu1[:,10:13]

#%% [markdown]

# ### vicon (vi*.csv)
#  Time  Header  translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w

# ### Sensors (imu*.csv)
# Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)

#%%







#%%



model = tf.keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), 
                               input_shape=(200, 6)))
model.add(layers.Dense(2))

model.summary()



model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])




# %%
