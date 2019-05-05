"""
This program loads all the resized data into an hdf5 file that is ready to use by the classifier (hdf5 stores np arrays).
Each signal is a vector of 300 values. The labels vary in length.
"""

import h5py
import random
import numpy as np

# Open the file.
input_label_file = h5py.File('ga10000_data_ready_for_training.hdf5', 'r')
num_training_values = 1
for key in list(input_label_file.keys()):
	num_training_values += len(list(input_label_file[key].keys()))


# Determine index ranges for the train, eval, and test data.
# Recall that some_array[x:y] includes the x value, but not the y value.
train_start_index = 0
train_end_index = train_start_index + int(num_training_values * 0.70)

eval_start_index = train_end_index
eval_end_index = eval_start_index + int(num_training_values * 0.15)

test_start_index = eval_end_index
test_end_index = num_training_values

# Randomly split the data into training, evaluation, and testing datasets.
randomized_index_array = [x for x in range (0, num_training_values)]
random.shuffle(randomized_index_array)
testing_indices = randomized_index_array[train_start_index : train_end_index]
evaluation_indices = randomized_index_array[eval_start_index : eval_end_index]
training_incices = randomized_index_array[test_start_index : test_end_index]


print("DONE SETTING ARRAY INDICES")


# Store the train, test, and eval data in a numpy array, using this function.
# NOTE: You have to zero pad the variable length output values. The CTC_Batch_Cost loss function handles this.


print("STARTING TO LOAD THE INPUTS")
# Load the input data. Also, keep track of the maximum label length (we need this to zero pad the labels).
inputs = np.empty([num_training_values, 300])
max_label_length = 0
input_count = 0
for key in list(input_label_file.keys()):
	for instance in input_label_file[key]:
		inputs[input_count] = input_label_file[key][instance]["Raw_Signal_Input"]
		label_len = len(list(input_label_file[key][instance]["Base_Pair_Label"][()]))
		if label_len > max_label_length:
			max_label_length = label_len
		input_count += 1
		print("done loading input: %d" % input_count)

print("DONE LOADING THE INPUTS")
print("STARTING TO LOAD THE LABELS")


# Load the labels. Also, keep track of each labels true length so that we can feed it to the CTC decoder.
labels = np.empty([num_training_values, max_label_length])
label_lengths = np.empty([num_training_values])
label_count = 0
for key in list(input_label_file.keys()):
	for instance in input_label_file[key]:
		# list() gives integer values (I think).
		# CHECK HERE FOR ISSUE WITH OUTPUT DATA TYPES
		the_label_unpadded = list(input_label_file[key][instance]["Base_Pair_Label"][()])
		unpadded_label_length = len(the_label_unpadded)
		# zero pad the labels and add them to the labels np array.
		labels[label_count] = np.pad(the_label_unpadded, (0, unpadded_label_length - max_label_length), "constant")
		label_lengths[label_count] = unpadded_label_length
		label_count += 1
print("DONE LOADING THE LABELS")


# Save these values to an hdf5 file.
train_eval_test_file = h5py.File("ga_4000_training_eval_test.hdf5", "w")
training = train_eval_test_file.create_group("Training")
evaluation = train_eval_test_file.create_group("Evaluation")
testing = train_eval_test_file.create_group("Testing")

training.create_dataset("Inputs", training_inputs.shape, data = training_inputs)
training.create_dataset("Labels", training_labels.shape, data = training_labels)
training.create_dataset("Label_Sizes", training_label_sizes.shape, data = training_label_sizes)

evaluation.create_dataset("Inputs", evaluation_inputs.shape, data = evaluation_inputs)
evaluation.create_dataset("Labels", evaluation_labels.shape, data = evaluation_labels)
evaluation.create_dataset("Label_Sizes", evaluation_label_sizes.shape, data = evaluation_label_sizes)

testing.create_dataset("Inputs", testing_inputs.shape, data = testing_inputs)
testing.create_dataset("Labels", testing_labels.shape, data = testing_labels)
testing.create_dataset("Label_Sizes", testing_label_sizes.shape, data = testing_label_sizes)


