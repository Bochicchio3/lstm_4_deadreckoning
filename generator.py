"""
On this file we have all the utility for managing data
in particular we build a data generator in order to
load the json one peace at time in order to be able to
compute the training even without more than 16 GB of RAM available
"""

import os
import numpy as np
import tensorflow as tf
import dataset_utils


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory_path, batch_size, windows_length = 200):
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
        self.Allfiles = os.listdir(directory_path) #list of all the files from the directory
        self.files = self.Allfiles.copy()
        print("\n\nthe file we will use for generator are: {}\n\n".format(self.files))
        self.current_file = self.files.pop()
        print(self.current_file)
        self.path = directory_path
        self.batch_size = batch_size

        self.data = tf.keras.utils.get_file(directory_path + self.current_file)
        """
        CSV for the dataset are
        
        input: [rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) user_acc_x(G) user_acc_y(G) user_acc_z(G)]
        ground truth: [translation.x translation.y rotation.z ]
        
        
        vicon (vi*.csv)
        Time  Header  translation.x translation.y translation.z rotation.x rotation.y rotation.z rotation.w


        Sensors (imu*.csv)
        Time attitude_roll(radians) attitude_pitch(radians) attitude_yaw(radians) rotation_rate_x(radians/s) rotation_rate_y(radians/s) rotation_rate_z(radians/s) gravity_x(G) gravity_y(G) gravity_z(G) user_acc_x(G) user_acc_y(G) user_acc_z(G) magnetic_field_x(microteslas) magnetic_field_y(microteslas) magnetic_field_z(microteslas)
       
        """






    def num_files(self):
        return len(self.Allfiles)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input['attention_mask']) / self.batch_size))

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
        # change the current file and add it to the done list
        if not self.files:
            self.files = self.Allfiles
        self.namefile = self.files.pop()


        # update the input and output tensors for this epoch

