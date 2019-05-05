import h5py
import math

# Open the hdf5 file that holds the sample of roughly 10000 {reads, signals}.
read_signal_file = h5py.File('ga10000_input_label_pairs.hdf5', 'r')


# Iterate through every entry in the ga10000_input_label_pairs.hdf5 file.
for key in list(read_signal_file.keys()):
	bp_to_signal_ratio = float(len(read_signal_file[key]['Base_Pair_Label'][()].decode("ascii"))) / float(read_signal_file[key]['Raw_Signal_Input'].shape[0])
	num_basepairs_per_300_signals = floor(300.0 * bp_to_signal_ratio)
	
	# Read the signal into a numpy array.
	read_signal_file[key]["Raw_Signal_Input"]

