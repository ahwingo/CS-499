"""
This program loads all the resized data into an hdf5 file that is ready to use by the classifier (hdf5 stores np arrays).
Each signal is a vector of 300 values. The labels vary in length.
"""

import h5py
import random
import numpy as np

# Open the file. Store a list of its keys.
input_label_file = h5py.File('ga_4000_input_label_pairs_resized_redone_the_faster_way_without_zeros_in_label_length.hdf5', 'r')

# Loop through the groups to get a full list of key tuples {key_1, key_2} and a count for the number of dataset values.
list_of_keys = []
for key_1 in list(input_label_file.keys()):
	for key_2 in list(input_label_file[key_1].keys()):
		list_of_keys.append((key_1, key_2))
num_dataset_values = len(list_of_keys)


# Determine index ranges for the train, eval, and test data.
# Recall that some_array[x:y] includes the x value, but not the y value.
train_start_index = 0
train_end_index = train_start_index + int(num_dataset_values * 0.70)

eval_start_index = train_end_index
eval_end_index = eval_start_index + int(num_dataset_values * 0.15)

test_start_index = eval_end_index
test_end_index = num_dataset_values

# Randomly split the data into training, evaluation, and testing datasets.
randomized_index_array = [x for x in range (0, num_dataset_values)]
random.shuffle(randomized_index_array)
training_indices = randomized_index_array[train_start_index : train_end_index]
evaluation_indices = randomized_index_array[eval_start_index : eval_end_index]
testing_indices = randomized_index_array[test_start_index : test_end_index]


print("DONE SETTING ARRAY INDICES")


# Store the train, test, and eval data in a numpy array, using this function.
# 	NOTE: You have to zero pad the variable length output values. The CTC_Batch_Cost loss function handles this.


# Load the training inputs, labels, and label lengths.
print("Working on training data.")
input_count = 0
max_label_length = 0
num_training_values = len(training_indices)
training_inputs = np.empty([num_training_values, 300])
for index in training_indices:
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	training_inputs[input_count] = input_label_file[key_1][key_2]["Raw_Signal_Input"]	
	label_len = len(list(input_label_file[key_1][key_2]["Base_Pair_Label"][()]))
	if label_len > max_label_length:
		max_label_length = label_len
	input_count += 1

label_count = 0
training_label_lengths = np.empty([num_training_values])
training_labels = np.empty([num_training_values, max_label_length])
for index in training_indices:
	key = list_of_keys[index]
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	the_label_unpadded = list(input_label_file[key_1][key_2]["Base_Pair_Label"][()])
	unpadded_label_length = len(the_label_unpadded)
	training_labels[label_count] = np.pad(the_label_unpadded, (0, max_label_length - unpadded_label_length), "constant")
	training_label_lengths[label_count] = unpadded_label_length
	label_count += 1

print("Done working on training data.")
print("Working on evaluation data.")
# Load the evaluation inputs, labels, and label lengths.
input_count = 0
max_label_length = 0
num_evaluation_values = len(evaluation_indices)
evaluation_inputs = np.empty([num_evaluation_values, 300])
for index in evaluation_indices:
	key = list_of_keys[index]
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	evaluation_inputs[input_count] = input_label_file[key_1][key_2]["Raw_Signal_Input"]	
	label_len = len(list(input_label_file[key_1][key_2]["Base_Pair_Label"][()]))
	if label_len > max_label_length:
		max_label_length = label_len
	input_count += 1

label_count = 0
evaluation_label_lengths = np.empty([num_evaluation_values])
evaluation_labels = np.empty([num_evaluation_values, max_label_length])
for index in evaluation_indices:
	key = list_of_keys[index]
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	the_label_unpadded = list(input_label_file[key_1][key_2]["Base_Pair_Label"][()])
	unpadded_label_length = len(the_label_unpadded)
	evaluation_labels[label_count] = np.pad(the_label_unpadded, (0, max_label_length - unpadded_label_length), "constant")
	evaluation_label_lengths[label_count] = unpadded_label_length
	label_count += 1


print("Done working on evaluation data.")
print("Working on testing data.")
# Load the testing inputs, labels, and label lengths.
input_count = 0
max_label_length = 0
num_testing_values = len(testing_indices)
testing_inputs = np.empty([num_testing_values, 300])
for index in testing_indices:
	key = list_of_keys[index]
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	testing_inputs[input_count] = input_label_file[key_1][key_2]["Raw_Signal_Input"]	
	label_len = len(list(input_label_file[key_1][key_2]["Base_Pair_Label"][()]))
	if label_len > max_label_length:
		max_label_length = label_len
	input_count += 1

label_count = 0
testing_label_lengths = np.empty([num_testing_values])
testing_labels = np.empty([num_testing_values, max_label_length])
for index in testing_indices:
	key = list_of_keys[index]
	key_1 = list_of_keys[index][0]
	key_2 = list_of_keys[index][1]
	the_label_unpadded = list(input_label_file[key_1][key_2]["Base_Pair_Label"][()])
	unpadded_label_length = len(the_label_unpadded)
	testing_labels[label_count] = np.pad(the_label_unpadded, (0, max_label_length - unpadded_label_length), "constant")
	testing_label_lengths[label_count] = unpadded_label_length
	label_count += 1



print("Done working on testing data.")
print("Printing everything to a file.")

# Save these values to an hdf5 file.
train_eval_test_file = h5py.File("ga_4000_training_eval_test_without_zeros.hdf5", "w")
training = train_eval_test_file.create_group("Training")
evaluation = train_eval_test_file.create_group("Evaluation")
testing = train_eval_test_file.create_group("Testing")

training.create_dataset("Inputs", training_inputs.shape, data = training_inputs)
training.create_dataset("Labels", training_labels.shape, data = training_labels)
training.create_dataset("Label_Sizes", training_label_lengths.shape, data = training_label_lengths)

evaluation.create_dataset("Inputs", evaluation_inputs.shape, data = evaluation_inputs)
evaluation.create_dataset("Labels", evaluation_labels.shape, data = evaluation_labels)
evaluation.create_dataset("Label_Sizes", evaluation_label_lengths.shape, data = evaluation_label_lengths)

testing.create_dataset("Inputs", testing_inputs.shape, data = testing_inputs)
testing.create_dataset("Labels", testing_labels.shape, data = testing_labels)
testing.create_dataset("Label_Sizes", testing_label_lengths.shape, data = testing_label_lengths)


