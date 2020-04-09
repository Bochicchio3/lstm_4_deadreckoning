import argparse
import tensorflow as tf
import generator

class PathPrediction(tf.keras.Model):

    """
    input: [input_size, batch_size, input_dimention, recurrent_hidden_size, rec_dropout, recurrent_layer]
    """
    
    def __init__(self,
                input_size,
                batch_size,
                input_dimention,
                recurrent_hidden_size,
                rec_dropout,
                recurrent_layer = tf.keras.layers.LSTM,

                ):
        super().__init__(name = "PathPrediction")

        self.input_size = input_size
        self.input = tf.keras.layers.Input(batch_shape = (batch_size, None, input_dimention), name = "encoder_input" )

        self.LSTM_layer = tf.keras.layers.Bidirectional(recurrent_layer(recurrent_hidden_size, return_sequences=True,
            recurrent_dropout=rec_dropout, name = "LSTM"), merge_mode='concat', name = "BI_LSTM")(encoder_input)

"""
LSTM:
input_size fissa

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    '''
    parser.add_argument('--model', type=str, default="LSTM", 
                help="The recurrent neural network you want to use: currently we support LSTM and GRU")

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    '''
    path = "Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/handheld"
    train_generator = generator.DataGenerator("Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/handheld", 100)
