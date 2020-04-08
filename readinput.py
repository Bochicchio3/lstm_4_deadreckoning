import os
import numpy as np
import glob


training_dir=['data1','data2','data3','data4']

testing_dir=['data5']

root='/home/alfredo/Desktop/PROJECTS/indoor_mapping/Oxford\ Inertial\ Odometry\ Dataset_2.0/Oxford\ Inertial\ Odometry\ Dataset/handheld/'

all_imu_data = []
all_vicon_data = []

for directory in training_dir:
    
    os.chdir(root+dir)
    imu_files=glob.glob('imu*')
    vicon_files=glob.glob('vi*')
    
    for file in imu_files:
        imu_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]
        
    for file in vicon_files:
        vicon_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]


def number_samples(all_imu_data):
    [for d. in all_imu_data] 