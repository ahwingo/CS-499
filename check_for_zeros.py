import h5py
f = h5py.File("ga_4000_input_label_pairs_resized_redone_the_faster_way.hdf5", "r")

for key_1 in list(f.keys()):
    for key_2 in list(f[key_1].keys()):
            if len(list(f[key_1][key_2]['Base_Pair_Label'][()])) == 0:
                    print("Fudge.... %s %s" % (key_1, key_2))
