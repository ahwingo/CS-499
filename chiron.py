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


def normalize_the_signals(the_signals):
    return minmax_scale(the_signals)


#-----------------------------------------------------------------------------------------------------------------------------------
#  Load the data.
#-----------------------------------------------------------------------------------------------------------------------------------
# Open the file that holds all of the data.
f = h5py.File("ga_4000_training_eval_test.hdf5", 'r')

# Load the training data.
training = f["Training"]
training_inputs = training["Inputs"]
len_training_inputs = training["Inputs"].shape[0]
training_inputs = normalize_the_signals(training_inputs)
training_inputs = np.reshape(training_inputs, (len_training_inputs, 300, 1))

training_labels = normalize_the_labels_to_values_between_0_and_1(training["Labels"])

print(training_labels.shape)
training_label_shapes = training["Label_Sizes"]


# Load the evaluation data.
evaluation = f["Evaluation"]
evaluation_inputs = evaluation["Inputs"]
num_evaluation_values = evaluation_inputs.shape[0]
evaluation_inputs = normalize_the_signals(evaluation_inputs)
evaluation_inputs = np.reshape(evaluation_inputs, (num_evaluation_values, 300, 1))
evaluation_labels = normalize_the_labels_to_values_between_0_and_1(evaluation["Labels"])
evaluation_label_shapes = evaluation["Label_Sizes"]

# Load the testing data.
testing = f["Testing"]
testing_inputs = testing["Inputs"]
testing_labels = testing["Labels"]
testing_label_shapes = testing["Label_Sizes"]


#-----------------------------------------------------------------------------------------------------------------------------------
#  Define a few network hyper parameters.
#-----------------------------------------------------------------------------------------------------------------------------------
batch_size = 400                      # In paper, batch size is 1100
num_timesteps_in_input_sequence = 300                   # Model input: a 300 dimensional vector holding a sequence of raw signals.
num_features_per_timestep = 1
num_recurrent_hidden_units = 100                  # With forward and backward cells, this becomes 200
num_fc_output_classes = 5                     # For the four nucleotides and one blank character.
input_lengths = np.full((len_training_inputs), 300)                 # This tensor is needed by the ctc_batch_cost function.
the_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  # As defined by the paper.



#-----------------------------------------------------------------------------------------------------------------------------------
#  Initialize model inputs. One of these inputs is the raw sequence.
#  The other two inputs hold information about the predicted / actual label lengths needed for the CTC decoder.
#-----------------------------------------------------------------------------------------------------------------------------------
#model_inputs = Input(batch_shape = (batch_size, num_timesteps_in_input_sequence, num_features_per_timestep))
#predicted_label_lengths = Input(batch_shape = (batch_size, 1))
#actual_label_lengths = Input(batch_shape = (batch_size, 1))


model_inputs = Input(shape = (num_timesteps_in_input_sequence, num_features_per_timestep))
predicted_label_lengths = Input(shape = (1,))
actual_label_lengths = Input(shape = (1,))

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
        print("\n\n\n\n YOU SUCK \n\n\n\n")
        #y_true = keras.backend.print_tensor(y_true, "\n\nY TRUE = ")
        #y_pred = keras.backend.print_tensor(y_pred, "Y PRED = ")
        the_cost = keras.backend.ctc_batch_cost(y_true, y_pred, pred_lab_lengths, act_lab_lengths)
        #the_cost = keras.backend.print_tensor(the_cost, "THE COST = ")
        return the_cost
    return loss



#-----------------------------------------------------------------------------------------------------------------------------------
# Initialize the model.   Compile the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
basecalling_model = Model(inputs = [model_inputs, predicted_label_lengths, actual_label_lengths], outputs = reshaped_fc_output)
#target_tensor = Input(batch_shape = (batch_size, training_labels.shape[1]))
target_tensor = Input(shape = (training_labels.shape[1],))
basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer, target_tensors = target_tensor)
#basecalling_model.compile(loss = custom_ctc_loss(predicted_label_lengths, actual_label_lengths), optimizer = the_optimizer)
print(basecalling_model.summary())


#-----------------------------------------------------------------------------------------------------------------------------------
#  Train the model.  
#-----------------------------------------------------------------------------------------------------------------------------------
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super(keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        rand_low = random.randint(0, num_evaluation_values - 32)        
        rand_high = rand_low + 32
        self.val_losses.append(self.model.evaluate([evaluation_inputs[rand_low:rand_high], input_lengths[rand_low:rand_high], evaluation_label_shapes[rand_low:rand_high]],
                                                    evaluation_labels[rand_low:rand_high], verbose=0))

loss_history = LossHistory()
#history = basecalling_model.fit(x = [training_inputs, input_lengths, training_label_shapes], 
#                                y = training_labels, batch_size = batch_size, epochs = 1, verbose = 1, shuffle = "batch", callbacks=[loss_history])

low_index = 0
high_index = 10000
history = basecalling_model.fit(x = [training_inputs[low_index:high_index], input_lengths[low_index:high_index], training_label_shapes[low_index:high_index]], 
                                y = training_labels[low_index:high_index], batch_size = batch_size, epochs = 3, verbose = 1, shuffle = "batch", callbacks=[loss_history])


print("TRAINING LOSSES: ")
print(loss_history.losses)


print("VALIDATION LOSSES: ")
print(loss_history.val_losses)

# Save the training and evaluation losses to an hdf5 file for later use.
losses_file = h5py.File("train_and_eval_losses_chiron.hdf5", "w")
train_loss_group = losses_file.create_group("Train_Losses")
train_loss_group.create_dataset("Losses", np.asarray(loss_history.losses).shape, data = np.asarray(loss_history.losses))
eval_loss_group = losses_file.create_group("Eval_Losses")
eval_loss_group.create_dataset("Losses", np.asarray(loss_history.val_losses).shape, data = np.asarray(loss_history.val_losses))

plt.plot(loss_history.losses)
plt.plot(loss_history.val_losses)
plt.title('Training and Validation Loss On the Chiron Model')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.legend(['Train', 'Eval'], loc='upper right')
plt.savefig('train_and_eval_loss_plot.png')

plt.clf()
plt.plot(loss_history.losses)
plt.title('Training Loss On the Chiron Model')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.savefig('train_loss_plot.png')


plt.clf()
plt.plot(loss_history.val_losses)
plt.title('Validation Loss On the Chiron Model')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.savefig('eval_loss_plot.png')

basecalling_model.save("trained_chiron_model.h5")

