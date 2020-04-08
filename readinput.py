import os
import numpy as np
import glob


class DataManager:
    def __init__(self):

        self.all_imu_data = []
        self.all_vicon_data = []

        self.training_dir=['data1','data2','data3','data4']
        self.testing_dir=['data5']

        root='/home/alfredo/Desktop/PROJECTS/indoor_mapping/Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/handheld/'

        self.load_data()

    def load_data(self):
        
        for directory in training_dir:
            
            os.chdir(root+directory)
            self.imu_files=glob.glob('imu*')
            self.vicon_files=glob.glob('vi*')

            for file in self.imu_files:
                imu_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]
                
            for file in vicon_files:
                vicon_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]


    def number_samples(self, batch_size):
        sum([d.shape[0]//batch_size for d in all_imu_data])