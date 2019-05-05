import h5py
import numpy as np
import threading

# This program uses X number of threads to get the data.
num_threads = 16

# Set the window length for the signal files.
signal_window_length = 300

# Open the hdf5 file that holds the sample of a few thousand {reads, signals}.
read_signal_file = h5py.File('ga_4000_input_label_pairs.hdf5', 'r')
num_input_label_pairs = len(list(read_signal_file.keys()))

# Write the resized training data to this hdf5 file.
input_label_pairs_file = h5py.File("ga_4000_input_label_pairs_ready_for_training.hdf5", "w")


def thread_function(low, high):


	print("Starting Thread where low = %d and high = %d" % (low, high))

	if high > num_input_label_pairs:
		high = num_input_label_pairs

	for key in list(read_signal_file.keys())[low:high]:

		# Create a group for this key in the output file.
		key_group = input_label_pairs_file.create_group(key)

		bp_to_signal_ratio = float(len(list(read_signal_file[key]['Base_Pair_Label'][()]))) / float(read_signal_file[key]['Raw_Signal_Input'].shape[0])
		num_basepairs_per_300_signals = int(300 * bp_to_signal_ratio)

		# Read the signal into a numpy array.
		full_length_raw_signal = read_signal_file[key]["Raw_Signal_Input"]

		# Read the label into a numpy array.
		full_length_bp_label = read_signal_file[key]['Base_Pair_Label'][()].decode("ascii")

	        # Iterate over the input signal and bp label.
		num_windows_in_signal = int(full_length_raw_signal.shape[0] / 300)
		counter = 0
		for i in range (0, num_windows_in_signal):

			print("In thread low: %d high: %d working on window %d / %d" % (low, high, i, num_windows_in_signal))

			# Add a new group to the output hdf5 file.
			read_chanel_instance_group = key_group.create_group("Instance_" + str(i))
		
			# Get the segment of the raw signal data.
			raw_signal_segment = full_length_raw_signal[300*i : 300*(i+1)]
			read_chanel_instance_group.create_dataset("Raw_Signal_Input", raw_signal_segment.shape, data = raw_signal_segment)	

			# Get the corresponding bp label.
			bp_label_segment = np.string_(full_length_bp_label[num_basepairs_per_300_signals*i : num_basepairs_per_300_signals*(i+1)])
			read_chanel_instance_group.create_dataset('Base_Pair_Label', bp_label_segment.shape, data = bp_label_segment)	
	



# Iterate through every entry in the ga10000_input_label_pairs.hdf5 file.
for i in range(0,num_threads):
	values_per_thread = int (num_input_label_pairs / num_threads)
	low = num_threads*i
	high = num_threads*(i+1)
	x = threading.Thread(target = thread_function, args = (low, high))
	x.start()
