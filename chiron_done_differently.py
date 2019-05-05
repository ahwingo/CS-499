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
from keras.optimizers import SGD

import h5py
import numpy as np


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  Set the neural network up, as defined in the Chiron paper (Input -> 5 CNN Blocks -> 3 RNN Blocks -> FC Layer -> CTC Decoder).
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

training_labels = training["Labels"]

print(training_labels.shape)
training_label_shapes = training["Label_Sizes"]


# Load the evaluation data.
evaluation = f["Evaluation"]
evaluation_inputs = evaluation["Inputs"]
evaluation_labels = evaluation["Labels"]
evaluation_label_shapes = evaluation["Label_Sizes"]

# Load the testing data.
testing = f["Testing"]
testing_inputs = testing["Inputs"]
testing_labels = testing["Labels"]
testing_label_shapes = testing["Label_Sizes"]


#-----------------------------------------------------------------------------------------------------------------------------------
#  Define a few network hyper parameters.
#-----------------------------------------------------------------------------------------------------------------------------------
batch_size = 10                      # In paper, batch size is 1100
num_timesteps_in_input_sequence = 300                   # Model input: a 300 dimensional vector holding a sequence of raw signals.
num_features_per_timestep = 1
num_recurrent_hidden_units = 100                  # With forward and backward cells, this becomes 200
num_fc_output_classes = 5                     # For the four nucleotides and one blank character.
input_lengths = np.full((len_training_inputs), 300)                 # This tensor is needed by the ctc_batch_cost function.
the_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  # As defined by the paper.
#the_optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) 


#-----------------------------------------------------------------------------------------------------------------------------------
#  Initialize model inputs. One of these inputs is the raw sequence.
#  The other two inputs hold information about the predicted / actual label lengths needed for the CTC decoder.
#  We also represent the output labels as a model input.
#-----------------------------------------------------------------------------------------------------------------------------------
model_inputs = Input(batch_shape = (batch_size, num_timesteps_in_input_sequence, num_features_per_timestep))
predicted_label_lengths = Input(batch_shape = (batch_size, 1))
actual_label_lengths = Input(batch_shape = (batch_size, 1))
labels = Input(batch_shape = (batch_size, training_labels.shape[1]))


#-----------------------------------------------------------------------------------------------------------------------------------
#  Next, feed the input into CNN block 1/5.
#-----------------------------------------------------------------------------------------------------------------------------------
# Branch 1 is the residual connection.
cnn_block_1_branch_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_1_branch_1")(model_inputs)

# Branch 2 is the stacked layers.
cnn_block_1_branch_2_part_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_1_branch_2_p1")(model_inputs)

cnn_block_1_branch_2_part_2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_1_branch_2_p2")(cnn_block_1_branch_2_part_1)

cnn_block_1_branch_2_part_3 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_1_branch_2_p3")(cnn_block_1_branch_2_part_2)

# Merge Branch 1 and 2. Apply a relu activation.
merged_cnn_block_1 = Add()([cnn_block_1_branch_1, cnn_block_1_branch_2_part_3])
activated_cnn_block_1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name="activated_cnn_block_1")(merged_cnn_block_1)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of CNN block 1/5 into CNN block 2/5.
#-----------------------------------------------------------------------------------------------------------------------------------
# Branch 1 is the residual connection.
cnn_block_2_branch_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_2_branch_1")(activated_cnn_block_1)

# Branch 2 is the stacked layers.
cnn_block_2_branch_2_part_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_2_branch_2_p1")(activated_cnn_block_1)

cnn_block_2_branch_2_part_2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_2_branch_2_p2")(cnn_block_2_branch_2_part_1)

cnn_block_2_branch_2_part_3 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_2_branch_2_p3")(cnn_block_2_branch_2_part_2)

# Merge Branch 1 and 2. Apply a relu activation.
merged_cnn_block_2 = Add()([cnn_block_2_branch_1, cnn_block_2_branch_2_part_3])
activated_cnn_block_2 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name="activated_cnn_block_2")(merged_cnn_block_2)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of CNN block 2/5 into CNN block 3/5.
#-----------------------------------------------------------------------------------------------------------------------------------
# Branch 1 is the residual connection.
cnn_block_3_branch_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_3_branch_1")(activated_cnn_block_2)

# Branch 2 is the stacked layers.
cnn_block_3_branch_2_part_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_3_branch_2_p1")(activated_cnn_block_2)

cnn_block_3_branch_2_part_2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_3_branch_2_p2")(cnn_block_3_branch_2_part_1)

cnn_block_3_branch_2_part_3 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_3_branch_2_p3")(cnn_block_3_branch_2_part_2)

# Merge Branch 1 and 2. Apply a relu activation.
merged_cnn_block_3 = Add()([cnn_block_3_branch_1, cnn_block_3_branch_2_part_3])
activated_cnn_block_3 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name="activated_cnn_block_3")(merged_cnn_block_3)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of CNN block 3/5 into CNN block 4/5.
#-----------------------------------------------------------------------------------------------------------------------------------
# Branch 1 is the residual connection.
cnn_block_4_branch_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_4_branch_1")(activated_cnn_block_3)

# Branch 2 is the stacked layers.
cnn_block_4_branch_2_part_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_4_branch_2_p1")(activated_cnn_block_3)

cnn_block_4_branch_2_part_2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_4_branch_2_p2")(cnn_block_4_branch_2_part_1)

cnn_block_4_branch_2_part_3 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_4_branch_2_p3")(cnn_block_4_branch_2_part_2)

# Merge Branch 1 and 2. Apply a relu activation.
merged_cnn_block_4 = Add()([cnn_block_4_branch_1, cnn_block_4_branch_2_part_3])
activated_cnn_block_4 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name="activated_cnn_block_4")(merged_cnn_block_4)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of CNN block 4/5 into CNN block 5/5.
# Final output has size [batch_size, num_timesteps_in_input_sequence, num_features_per_timestep]
#-----------------------------------------------------------------------------------------------------------------------------------
# Branch 1 is the residual connection.
cnn_block_5_branch_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_5_branch_1")(activated_cnn_block_4)

# Branch 2 is the stacked layers.
cnn_block_5_branch_2_part_1 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_5_branch_2_p1")(activated_cnn_block_4)

cnn_block_5_branch_2_part_2 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation='relu', use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_5_branch_2_p2")(cnn_block_5_branch_2_part_1)

cnn_block_5_branch_2_part_3 = Conv1D(filters=256, kernel_size=1, strides=1, padding='same', data_format='channels_last', 
            dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', 
                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="cnn_block_5_branch_2_p3")(cnn_block_5_branch_2_part_2)

# Merge Branch 1 and 2. Apply a relu activation.
merged_cnn_block_5 = Add()([cnn_block_5_branch_1, cnn_block_5_branch_2_part_3])
activated_cnn_block_5 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0, name="activated_cnn_block_5")(merged_cnn_block_5)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of CNN block 5/5 into RNN block 1/3.
#-----------------------------------------------------------------------------------------------------------------------------------
rnn_block_1_init = LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
          recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                      recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, 
                            return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)
rnn_block_1 = Bidirectional(rnn_block_1_init, merge_mode='concat', weights=None, name="rnn_block_1")(activated_cnn_block_5)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of RNN block 1/3 into RNN block 2/3.
#-----------------------------------------------------------------------------------------------------------------------------------
rnn_block_2_init = LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
          recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                      recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, 
                            return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)
rnn_block_2 = Bidirectional(rnn_block_2_init, merge_mode='concat', weights=None, name="rnn_block_2")(rnn_block_1)



#-----------------------------------------------------------------------------------------------------------------------------------
# Next, feed the output of RNN block 2/3 into RNN block 3/3.
# Final output has size [batch_size, num_timesteps_in_input_sequence, 200]
#-----------------------------------------------------------------------------------------------------------------------------------
rnn_block_3_init = LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
          recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                      recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, 
                            return_sequences=True, return_state=False, go_backwards=False, stateful=False, unroll=False)
rnn_block_3 = Bidirectional(rnn_block_3_init, merge_mode='concat', weights=None, name="rnn_block_3")(rnn_block_2)



#-----------------------------------------------------------------------------------------------------------------------------------
# Then, feed the (reshaped) output of RNN block 3/3 into a fully connected layer.
#-----------------------------------------------------------------------------------------------------------------------------------
#reshaped_rnn_output = Reshape((batch_size*num_timesteps_in_input_sequence, 200))(rnn_block_3)
reshaped_rnn_output = Reshape((num_timesteps_in_input_sequence, 200))(rnn_block_3)
#fully_connected_output = Dense(num_fc_output_classes, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
y_pred = Dense(num_fc_output_classes, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, bias_constraint=None, name="fully_connected_output")(reshaped_rnn_output)

x = keras.backend.print_tensor(y_pred, message='Y_Pred Values: ')



#-----------------------------------------------------------------------------------------------------------------------------------
#  Finally, reshape the output of the fully connected layer. With this shape, it CAN be fed to a CTC decoder.
#-----------------------------------------------------------------------------------------------------------------------------------

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


#y_pred = Reshape((num_timesteps_in_input_sequence, num_fc_output_classes))(fully_connected_output)
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, predicted_label_lengths, actual_label_lengths])


basecalling_model = Model(inputs = [model_inputs, labels, predicted_label_lengths, actual_label_lengths], outputs = loss_out)
basecalling_model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer = the_optimizer)
basecalling_model.metrics_tensors = [y_pred]



"""
def custom_ctc_loss(pred_lab_lengths, act_lab_lengths):
    def loss(y_true, y_pred):
        print("\n\n\n\n YOU SUCK \n\n\n\n")
        tf.keras.backend.print_tensor(y_true)
        tf.keras.backend.print_tensor(y_pred)
        return keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
    return loss



#-----------------------------------------------------------------------------------------------------------------------------------
# Initialize the model.   Compile the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
basecalling_model = Model(inputs = [model_inputs, predicted_label_lengths, actual_label_lengths], outputs = reshaped_fc_output)
target_tensor = Input(batch_shape = (batch_size, training_labels.shape[1]))
basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
#basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer)
"""



print(basecalling_model.summary())


dummy_outputs = np.zeros([10000])

low_index = 0
high_index = 100
basecalling_model.fit(x = [training_inputs[low_index:high_index], training_labels[low_index:high_index], 
                            input_lengths[low_index:high_index], training_label_shapes[low_index:high_index]], 
                            y = dummy_outputs[low_index:high_index], batch_size = batch_size, epochs = 1, verbose = 1, shuffle = "batch")


low_index = 200
high_index = 300
basecalling_model.fit(x = [training_inputs[low_index:high_index], training_labels[low_index:high_index], 
                            input_lengths[low_index:high_index], training_label_shapes[low_index:high_index]], 
                            y = dummy_outputs[low_index:high_index], batch_size = batch_size, epochs = 1, verbose = 1, shuffle = "batch")




"""
#-----------------------------------------------------------------------------------------------------------------------------------
#  Train the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
for batch_range in range(0, 1):
    low_index = batch_range*100
    high_index = (batch_range+1)*100
    basecalling_model.fit(x = [training_inputs[low_index:high_index], input_lengths[low_index:high_index], training_label_shapes[low_index:high_index]], 
                            y = training_labels[low_index:high_index], batch_size = batch_size, epochs = 1, verbose = 1, shuffle = "batch")


""" 
