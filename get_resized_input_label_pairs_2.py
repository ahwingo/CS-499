# NOTE: HDF5 users report extreme slowdowns when storing many datasets in the same group.
#	To avoid this, store the data in buckets.

import h5py
import numpy as np

# Set the window length for the signal files.
signal_window_length = 300

# Open the hdf5 file that holds the sample of roughly 10000 {reads, signals}.
read_signal_file = h5py.File('ga_4000_input_label_pairs.hdf5', 'r')

# Write the resized training data to this hdf5 file.
input_label_pairs_file = h5py.File("ga_4000_input_label_pairs_resized_redone_the_faster_way_without_zeros_in_label_length.hdf5", "w")


# Iterate through every entry in the ga10000_input_label_pairs.hdf5 file.
counter = 1
total_num = len(list(read_signal_file.keys()))
for key in list(read_signal_file.keys()):

	print("Working on file " + str(counter) + " of " + str(total_num))
	counter += 1

	# Create a group for this key in the output file.	
	key_group = input_label_pairs_file.create_group(key)

	bp_to_signal_ratio = float(len(read_signal_file[key]['Base_Pair_Label'][()].decode("ascii"))) / float(read_signal_file[key]['Raw_Signal_Input'].shape[0])
	num_basepairs_per_300_signals = int(300 * bp_to_signal_ratio)

	# If this happens, it will really mess up our data. Just skip these instances.
	if (num_basepairs_per_300_signals == 0):
		continue

	# Read the signal into a numpy array.
	full_length_raw_signal = read_signal_file[key]["Raw_Signal_Input"]

	# Read the label into a numpy array.
	full_length_bp_label = read_signal_file[key]['Base_Pair_Label'][()].decode("ascii")

        # Iterate over the input signal and bp label.
	num_windows_in_signal = int(full_length_raw_signal.shape[0] / 300)
	print("num_windows_in_signal = " + str(num_windows_in_signal))
	for i in range (0, num_windows_in_signal):

		# Add a new group to the output hdf5 file.
		read_chanel_instance_group = key_group.create_group("Instance_" + str(i))
		
		# Get the segment of the raw signal data.
		raw_signal_segment = full_length_raw_signal[300*i : 300*(i+1)]
		read_chanel_instance_group.create_dataset("Raw_Signal_Input", raw_signal_segment.shape, data = raw_signal_segment)	

		# Get the corresponding bp label.
		bp_label_segment = np.string_(full_length_bp_label[num_basepairs_per_300_signals*i : num_basepairs_per_300_signals*(i+1)])
		read_chanel_instance_group.create_dataset('Base_Pair_Label', bp_label_segment.shape, data = bp_label_segment)	
