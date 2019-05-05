import h5py
import numpy as np

# Set the window length for the signal files.
signal_window_length = 300

# Open the hdf5 file that holds the sample of roughly 10000 {reads, signals}.
read_signal_file = h5py.File('ga10000_input_label_pairs.hdf5', 'r')

total_num = 0
for key in list(read_signal_file.keys()):
	total_num += read_signal_file[key]['Raw_Signal_Input'].shape[0] / 300
print(total_num)
