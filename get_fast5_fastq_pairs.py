"""
/jlf/jmsutton2/DSM44187-Jan9/GA10000/reads/0


GXB01119_20180110__GA10000_sequencing_run_DSM44187_Jan9_21769_read_67_ch_381_strand.fast5


filename	read_id	run_id	channel	start_time	duration	num_events	template_start	num_events_template	template_duration	num_called_template	sequence_length_template	mean_qscore_template	strand_score_template
GXB01119_20180110__GA10000_sequencing_run_DSM44187_Jan9_21769_read_23749_ch_470_strand.fast5	903d3ba8-6e75-4091-bd34-e8a0749961b9	54ecfc44944bc5acbc478ebdc3b12de3c37cff3d	470	49940.20275	28.6975	22752	0.02325	22752	28.67425	22752	10181	10.871	-0.0013



@508790da-3975-4bcd-b953-3475d681c64a runid=54ecfc44944bc5acbc478ebdc3b12de3c37cff3d device_id=GA10000 read=15044 ch=153 start_time=2018-01-11T04:10:47Z

@f3e94b14-925c-4484-8ce9-8c60cf71ddc3 runid=54ecfc44944bc5acbc478ebdc3b12de3c37cff3d device_id=GA10000 read=6580 ch=493 start_time=2018-01-10T06:56:28Z



"""


from os import listdir
from os.path import isfile, join
from itertools import islice
from itertools import zip_longest
import numpy as np
import h5py
import random


# This function lets us read the fastq file four lines at a time.
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


# Open a hdf5 file for saving our data ( {input, lable} pairs ) to.
input_label_pairs_file = h5py.File("ga10000_input_label_pairs.hdf5", "w")


# Build a list of all fastq files. We will read through these.
fastq_directory = "/jlf/jmsutton2/DSM44187-Jan9/GA10000"
fastq_files = [fastq_directory + "/" + f for f in listdir(fastq_directory) if isfile(join(fastq_directory, f)) and f.endswith("fastq")]

# Use this dictionary to store all fast5 files for near constant lookup times.
fast5_files = {}
#for directory_number in range(0, 280):
for directory_number in range(0, 288):
	fast5_directory = "/jlf/jmsutton2/DSM44187-Jan9/GA10000/reads/" + str(directory_number)
	for f in listdir(fast5_directory):
		if isfile(join(fast5_directory, f)) and f.endswith("fast5"):
			# use the directory number as the value. this lets us find the file again.
			fast5_files[f] = directory_number
		

# We want to collect 10224 reads, sampled randomly from the various fastq files.
# Let each of the 284 fastq file equally contribute 36 reads. 
# For each fastq file, we want to open it, and read line by line to identify corresponding fast5 files.
# When a match is found, create a new hdf5 dataset, to add to input_label_pairs_file, which will hold the signals and uncleaned labels (they can be cleaned later).
for fastq in fastq_files:

	print("Collecting samples from file: " + fastq)

	opened_fastq = open(fastq, "r")
	# If the file is the /jlf/jmsutton2/DSM44187-Jan9/GA10000/fastq_283.fastq, max number of reads is 3463.
	# Otherwise, max number of reads is 4000.
	max_num_reads_in_file = 4000
	if (fastq == "/jlf/jmsutton2/DSM44187-Jan9/GA10000/fastq_283.fastq"):
		max_num_reads_in_file = 3463
	# This list will determine which reads are randomly sampled for each file.
	random_sampler = random.sample(range(0, max_num_reads_in_file), 36)
	
	counter = -1
	for lines in grouper(opened_fastq, 4):

		counter += 1
		if counter not in random_sampler:
			continue
		
		assert len(lines) == 4
		# The first line holds the description for the read. from this, we can find the fast5 file, if it exists.
		read_description = lines[0].rstrip()
		read_number = read_description[(read_description.find("read=") + 5) : read_description.find(" ch=")]
		chanel_number = read_description[(read_description.find(" ch=") + 4) : read_description.find(" start_time=")]
			
		# Search the dictionary of fast5 files for this data.
		# GXB01119_20180110__GA10000_sequencing_run_DSM44187_Jan9_21769_read_67_ch_381_strand.fast5
		possible_fast5_file = "GXB01119_20180110__GA10000_sequencing_run_DSM44187_Jan9_21769_read_" + read_number + "_ch_" + chanel_number + "_strand.fast5"

		# Store the base pair labels as a string.
		base_pair_labels = read_description = lines[1].rstrip()
		base_pair_labels_as_np_array = np.array(list(base_pair_labels))
		base_pair_labels_as_np_string = np.string_(base_pair_labels)

		# Also for later, store the fastq score.
		fastq_score = np.string_(list(lines[3].rstrip()))
		

		if possible_fast5_file in fast5_files:
			# Get the path to the fast5 file so that we can open it.
			fast5_dir = "/jlf/jmsutton2/DSM44187-Jan9/GA10000/reads/" + str(fast5_files[possible_fast5_file]) + "/"
			fast5_path = fast5_dir + possible_fast5_file	
	
			# Open the found fast5 file to get the raw signal.
			raw_signal_file = h5py.File(fast5_path, 'r')
			raw_signal = raw_signal_file["Raw"]["Reads"]["Read_" + read_number]["Signal"]

			# Create a new group in the input_label_pairs_file to add the signal and the label
			read_chanel_group = input_label_pairs_file.create_group("Read_Chanel_" + read_number + "_" + chanel_number)
			read_chanel_group.create_dataset("Raw_Signal_Input", raw_signal.shape, data = raw_signal)
			#read_chanel_group.create_dataset("Base_Pair_Label", base_pair_labels_as_np_array.shape, data = base_pair_labels_as_np_array)
			read_chanel_group.create_dataset("Base_Pair_Label", base_pair_labels_as_np_string.shape, data = base_pair_labels_as_np_string)
			read_chanel_group.create_dataset("FastQ_Score", fastq_score.shape, data = fastq_score)
			
		else:
			print("UNABLE TO FIND MATCH FOR possible_fast5_file: " + possible_fast5_file)
