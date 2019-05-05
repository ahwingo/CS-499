import tensorflow as tf

import keras
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.layers import ReLU
from keras.layers import Add
from keras.layers import Lambda

import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale


import matplotlib.pyplot as plt
import random

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  Set the neural network up, as defined in the Chiron paper (Input -> 5 CNN Blocks -> 3 RNN Blocks -> FC Layer -> CTC Decoder).
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def normalize_the_labels_to_values_between_0_and_1(the_labels):
        #return minmax_scale(the_labels)
        copy_of_the_labels = np.copy(the_labels)
        level_1 = 0
        for arr in copy_of_the_labels:
            level_2 = 0
            for x in arr:
                if x == 65:
                    copy_of_the_labels[level_1][level_2] = 0
                elif x == 67:
                    copy_of_the_labels[level_1][level_2] = 1
                elif x == 71:
                    copy_of_the_labels[level_1][level_2] = 2
                elif x == 84:
                    copy_of_the_labels[level_1][level_2] = 3
                level_2 += 1
            level_1 += 1
        return copy_of_the_labels
"""
#-----------------------------------------------------------------------------------------------------------------------------------
#  Load the data.
#-----------------------------------------------------------------------------------------------------------------------------------
# Open the file that holds all of the data.
f = h5py.File("ga_4000_training_eval_test.hdf5", 'r')

# Load the training data.
training = f["Training"]
training_inputs = training["Inputs"]
len_training_inputs = training["Inputs"].shape[0]
training_inputs = np.reshape(training_inputs, (len_training_inputs, 300, 1))

training_labels = normalize_the_labels_to_values_between_0_and_1(training["Labels"])

print(training_labels.shape)
training_label_shapes = training["Label_Sizes"]

input_lengths = np.full((len_training_inputs), 300)                 # This tensor is needed by the ctc_batch_cost function.

# Load the evaluation data.
evaluation = f["Evaluation"]
evaluation_inputs = evaluation["Inputs"]
num_evaluation_values = evaluation_inputs.shape[0]
evaluation_inputs = np.reshape(evaluation_inputs, (num_evaluation_values, 300, 1))
evaluation_labels = normalize_the_labels_to_values_between_0_and_1(evaluation["Labels"])
evaluation_label_shapes = evaluation["Label_Sizes"]

# Load the testing data.
testing = f["Testing"]
testing_inputs = testing["Inputs"]
testing_labels = testing["Labels"]
testing_label_shapes = testing["Label_Sizes"]
"""

#-----------------------------------------------------------------------------------------------------------------------------------
#  Define a few network hyper parameters.
#-----------------------------------------------------------------------------------------------------------------------------------
batch_size = 300                      # In paper, batch size is 1100
num_timesteps_in_input_sequence = 300                   # Model input: a 300 dimensional vector holding a sequence of raw signals.
num_recurrent_hidden_units = 100                  # With forward and backward cells, this becomes 200
num_fc_output_classes = 5                     # For the four nucleotides and one blank character.
the_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  # As defined by the paper.



#-----------------------------------------------------------------------------------------------------------------------------------
#  Initialize model inputs. One of these inputs is the raw sequence.
#  The other two inputs hold information about the predicted / actual label lengths needed for the CTC decoder.
#-----------------------------------------------------------------------------------------------------------------------------------
#model_inputs = Input(batch_shape = (batch_size, num_timesteps_in_input_sequence, num_features_per_timestep))
#predicted_label_lengths = Input(batch_shape = (batch_size, 1))
#actual_label_lengths = Input(batch_shape = (batch_size, 1))
model_inputs = Input(shape = (num_timesteps_in_input_sequence, 1))
predicted_label_lengths = Input(shape = (1,))
actual_label_lengths = Input(shape = (1,))



#-----------------------------------------------------------------------------------------------------------------------------------
#  Next, feed the input into 72 Residual Blocks.
#-----------------------------------------------------------------------------------------------------------------------------------


activated_one = Activation('elu')(model_inputs)
batch_norm_one = BatchNormalization(axis=1)(activated_one)
conv_one = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(batch_norm_one)
activated_two = Activation('elu')(conv_one)
batch_norm_two = BatchNormalization(axis=1)(activated_two)
conv_two = Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(batch_norm_two)
res_sum = Add()([conv_two, model_inputs])

#reshaped_res_output = Reshape((64,))(res_sum)
softmax_layer = Dense(5, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, bias_constraint=None, name="fully_connected_output")(res_sum)

reshaped_fc_output = Reshape((300, 5))(softmax_layer)



"""

def custom_ctc_loss(pred_lab_lengths, act_lab_lengths):
    def loss(y_true, y_pred):
        print("\n\n\n\n YOU SUCK \n\n\n\n")
        #y_true = keras.backend.print_tensor(y_true, "\n\nY TRUE = ")
        #y_pred = keras.backend.print_tensor(y_pred, "Y PRED = ")
        the_cost = keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
        #the_cost = keras.backend.print_tensor(the_cost, "THE COST = ")
        return the_cost
    return loss

basecalling_model = Model(inputs = [model_inputs, predicted_label_lengths, actual_label_lengths], outputs = reshaped_fc_output)
target_tensor = Input(shape = (training_labels.shape[1],))
basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
"""
