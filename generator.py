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
    def __init__(self, directory_path, batch_size, window_length = 200):
        
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
        
        self.windows_length=windows_length
        self.batch_size = batch_size
        
        self.all_imu_data = []
        self.all_vicon_data = []

        self.training_dir=['data1','data2','data3','data4']
        self.testing_dir=['data5']

        root='/home/alfredo/Desktop/PROJECTS/indoor_mapping/Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/handheld/'

        self.load_data()

        # number of sequences divided by number of batches
        self.number_sequences = sum([d.shape[0]//window_length for d in self.all_imu_data])

    def preprocess(self):
        # for all the file 
        self.windows_input = []
        self.windows_output = []
        for imu, vicon in zip(self.imu_data, self.vicon_data):
            # take the windows
            num_windows = imu.shape[0]//self.window_length
            starting_point = random.randint(0, imu.shape[0]%self.window_length)

            imu_copy = imu.copy()[starting_point:, :]
            vicon_copy = vicon.copy()[starting_point:, :]

            splitting = [self.window_length]





    def load_data(self):
        
        for directory in training_dir:
            
            os.chdir(root+directory)
            self.imu_files=glob.glob('imu*')
            self.vicon_files=glob.glob('vi*')

            for file in self.imu_files:
                imu_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]
                
            for file in vicon_files:
                vicon_data = np.genfromtxt(os.path.join(directory, file), delimiter=',')[:,2:]




    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.number_sequences//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        '''
        this is the dictionary names for the input and output of the model
         x = {
            'input_ids':
            'attention_mask':
            'token_type_ids':
        }
        y = {
            'start':
            'end':
            'type':
        }
        '''
        x = {k:v[self.batch_size*index:self.batch_size*(index+1)] for k, v in self.input.items()}


        y = [v[self.batch_size*index:self.batch_size*(index+1)] for v in self.output]


        return x, y

    def on_epoch_end(self):
        self.last_index = 0

