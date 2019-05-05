import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):

    def __init__(self, all_inputs, all_input_shapes, all_labels, all_label_shapes, batch_size=32):
        self.batch_size = batch_size
        self.all_inputs = all_inputs
        self.all_input_shapes = all_input_shapes
        self.all_labels = all_labels
        self.all_label_shapes = all_label_shapes
        self.current_batch_number = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(all_inputs.shape[0] / self.batch_size))

    def __getitem__(self, index):
        # Generate data
        X, y = self.__data_generation()
        self.current_batch_number += 1
        return X, y


    def on_epoch_end(self):
        self.current_batch_number = 0

    def __data_generation(self):
        low = self.batch_size * self.current_batch_number
        high = self.batch_size * (1 + self.current_batch_number)
        batch_of_inputs = self.all_inputs[low:high]
        batch_of_input_shapes = self.all_input_shapes[low:high]
        batch_of_labels = self.all_labels[low:high]
        batch_of_label_shapes = self.all_label_shapes[low:high]

        return batch_of_inputs, [batch_of_input_shapes, batch_of_label_shapes, batch_of_labels]



