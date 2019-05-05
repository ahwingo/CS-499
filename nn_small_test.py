import tensorflow as tf

import plaidml.keras
plaidml.keras.install_backend()

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
f = h5py.File("ga_4000_training_eval_test_without_zeros.hdf5", 'r')

# Load the training data.
training = f["Training"]
training_inputs = training["Inputs"]
len_training_inputs = training["Inputs"].shape[0]
training_inputs = np.reshape(training_inputs, (len_training_inputs, 300, 1))

training_labels = training["Labels"]
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
batch_size = 1												# In paper, batch size is 1100
num_timesteps_in_input_sequence = 300									# Model input: a 300 dimensional vector holding a sequence of raw signals.
num_features_per_timestep = 1
num_recurrent_hidden_units = 100									# With forward and backward cells, this becomes 200
num_fc_output_classes = 5										# For the four nucleotides and one blank character.
input_lengths = np.full((len_training_inputs), 300) 								# This tensor is needed by the ctc_batch_cost function.
the_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)	# As defined by the paper.



#-----------------------------------------------------------------------------------------------------------------------------------
#  Initialize model inputs. One of these inputs is the raw sequence.
#  The other two inputs hold information about the predicted / actual label lengths needed for the CTC decoder.
#-----------------------------------------------------------------------------------------------------------------------------------
model_inputs = Input(batch_shape = (batch_size, num_timesteps_in_input_sequence, num_features_per_timestep))
#model_inputs = Input(batch_shape = (batch_size, num_timesteps_in_input_sequence))
predicted_label_lengths = Input(batch_shape = (batch_size, 1))
actual_label_lengths = Input(batch_shape = (batch_size, 1))



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
fully_connected_output = Dense(num_fc_output_classes, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
				bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
				kernel_constraint=None, bias_constraint=None, name="fully_connected_output")(reshaped_rnn_output)



#-----------------------------------------------------------------------------------------------------------------------------------
#  Finally, reshape the output of the fully connected layer. With this shape, it CAN be fed to a CTC decoder.
#-----------------------------------------------------------------------------------------------------------------------------------
reshaped_fc_output = Reshape((num_timesteps_in_input_sequence, num_fc_output_classes))(fully_connected_output)


def custom_ctc_loss(pred_lab_lengths, act_lab_lengths):
	def loss(y_true, y_pred):
		return keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
	return loss



#-----------------------------------------------------------------------------------------------------------------------------------
# Initialize the model.   Compile the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
basecalling_model = Model(inputs = [model_inputs, predicted_label_lengths, actual_label_lengths], outputs = reshaped_fc_output)
target_tensor = Input(batch_shape = (batch_size, 1))
#target_tensor = Input(batch_shape = (5, 1))
basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
print(basecalling_model.summary())

"""
#-----------------------------------------------------------------------------------------------------------------------------------
#  This is the custom loss function (Keras doesn't really support CTC Loss).
#-----------------------------------------------------------------------------------------------------------------------------------
def custom_ctc_loss(act_lab_lengths):
	def loss(y_true, y_pred):
		pred_lab_lengths = input_lengths
		return keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
	return loss



#-----------------------------------------------------------------------------------------------------------------------------------
#  Initialize the model.   Compile the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
basecalling_model = Model(inputs = [model_inputs, actual_label_lengths], outputs = reshaped_fc_output)
target_tensor = Input(batch_shape = (batch_size, 1))
#basecalling_model.compile(loss = custom_ctc_loss(actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
basecalling_model.compile(loss = custom_ctc_loss(actual_label_lengths), optimizer = the_optimizer, target_tensors = None)
#basecalling_model.compile(loss = custom_ctc_loss(actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
print(basecalling_model.summary())
"""


#-----------------------------------------------------------------------------------------------------------------------------------
#  Train the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
basecalling_model.fit(x = [training_inputs[0:5], input_lengths[0:5], training_label_shapes[0:5]], y = training_labels[0:5], batch_size = batch_size, epochs = 100, verbose = 1, shuffle = "batch")





