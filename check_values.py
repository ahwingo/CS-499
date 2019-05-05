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




print(list(training_inputs[0]))
print(list(training_labels[0]))

print("\n\n\n\n")

print(list(training_inputs[2]))
print(list(training_labels[2]))
