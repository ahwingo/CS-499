import tensorflow as tf
import keras
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Dense
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.layers import ReLU
from keras.layers import Add
from keras.layers import Lambda

from keras.models import load_model

import h5py
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import random


# Load the test data. Normalize the inputs and labels appropriately.

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

def normalize_the_signals(the_signals):
    return minmax_scale(the_signals)


f = h5py.File("ga_4000_training_eval_test.hdf5", 'r')

testing = f["Testing"]
testing_inputs = testing["Inputs"]
len_testing_inputs = testing["Inputs"].shape[0]
testing_inputs = normalize_the_signals(testing_inputs)
testing_inputs = np.reshape(testing_inputs, (len_testing_inputs, 300, 1))
testing_labels = normalize_the_labels_to_values_between_0_and_1(testing["Labels"])
testing_label_shapes = testing["Label_Sizes"]
input_lengths = np.full((len_testing_inputs), 300)                 # This tensor is needed by the ctc_batch_cost function.


def custom_ctc_loss(pred_lab_lengths, act_lab_lengths):
    def loss(y_true, y_pred):
        print("\n\n\n\n YOU SUCK \n\n\n\n")
        #y_true = keras.backend.print_tensor(y_true, "\n\nY TRUE = ")
        #y_pred = keras.backend.print_tensor(y_pred, "Y PRED = ")
        the_cost = keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
        #the_cost = keras.backend.print_tensor(the_cost, "THE COST = ")
        return the_cost
    return loss

def loss_max(y_true, y_pred):
    from keras import backend as K
    return K.max(K.abs(y_pred - y_true), axis=-1)


model = load_model('trained_chiron_model_randnorm.h5', custom_objects={'loss': loss_max})


input_instance = np.array([testing_inputs[1], input_lengths[1], testing_label_shapes[1]])
#prediction = model.predict(input_instance)
prediction = model.predict([testing_inputs, input_lengths, testing_label_shapes], batch_size=1)
print(prediction)




