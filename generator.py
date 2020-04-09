"""
On this file we have all the utility for managing data
in particular we build a data generator in order to
load the json one peace at time in order to be able to
compute the training even without more than 16 GB of RAM available
"""


import os
import numpy as np
import tensorflow as tf
import random
import glob

"""
CSV for the dataset are

input: [rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) user_acc_x(G) user_acc_y(G) user_acc_z(G)]
ground truth: [translation.x translation.y rotation.z ]


vicon (vi*.csv)
Time  Header  translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w


Sensors (imu*.csv)
Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)

"""




class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory_path, batch_size, window_length = 200, validation = False):
        
        'Initialization'
        
        '''
        Load the files and create the question answer tuple
        store everithing

        @param batch_size integer for the size of the batch
        @param directory_path path of the directory containing the training files
        @param namemodel name of model to use (bert, albert)
        @param vocab the vocabulary
        @param max_num_samples integer for the maximum number of samples
        '''
        self.type = 'syn'

        self.windows_length=window_length
        self.batch_size = batch_size
        
        self.all_imu_data = []
        self.all_vicon_data = []

        self.path = "Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/handheld"
        self.training_dir=['data1','data2','data3','data4']
        self.testing_dir=['data5']

        if validation:
            self.load_data(self.testing_dir)
        else:
            self.load_data(self.training_dir)

        # number of sequences divided by number of batches
        self.number_sequences = sum([d.shape[0]//window_length for d in self.all_imu_data])
        self.from_files_to_windows()
        print("finish to load the data")

    def from_files_to_windows(self):
        # Function that takes all the data from the files and create a list of windows for the epoch 
        self.windows_input = []
        self.windows_output = []
        for imu, vicon in zip(self.all_imu_data, self.all_vicon_data):
            # take the windows
            num_windows = imu.shape[0]//self.windows_length
            chunks_to_delete = imu.shape[0]%self.windows_length
            starting_point = random.randint(0, imu.shape[0]%self.windows_length)

            if chunks_to_delete!= starting_point:
                imu_copy = imu.copy()[starting_point:- chunks_to_delete + starting_point, :]
                vicon_copy = vicon.copy()[starting_point:-chunks_to_delete +starting_point, :]
            else:
                # if I have to 
                imu_copy = imu.copy()[starting_point:, :]
                vicon_copy = vicon.copy()[starting_point:, :]


            self.windows_input += np.array_split(imu_copy, num_windows)

            self.windows_output += np.array_split(vicon_copy, num_windows)

            last_peace = imu_copy[-200:, :]
            assert((self.windows_input[-1] == last_peace).min())

            first_peace = imu_copy[:200, :]
            assert((self.windows_input[-num_windows] == first_peace).min())




    def load_data(self, directory_list):
    
        for directory in directory_list:

            imu_files = glob.glob(os.path.join(self.path,directory, self.type,  'imu*.csv'))
            vicon_files = glob.glob(os.path.join(self.path, directory, self.type, 'vi*.csv'))


            for file in imu_files:
                self.all_imu_data.append(np.genfromtxt(file, delimiter=',')[:,2:])
                index = file[-5]
                vicon_file = [v for v in vicon_files if v[-5] == file[-5]]
                self.all_vicon_data.append(np.genfromtxt(*vicon_file, delimiter=',')[:,2:])


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.number_sequences//self.batch_size

    def __getitem__(self, index):
        bs = self.batch_size
        return self.windows_input[bs*index:bs*(index + 1)], self.windows_output[bs*index:bs*(index + 1)]

    def on_epoch_end(self):
        self.from_files_to_windows()

