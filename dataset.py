import numpy as np
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
from common import config

class MnistDataloader(object):
    """Abstraction for loading the MNIST dataset."""
    def __init__(self):
        self.training_images_filepath = join(config("dataset_dir"), config("training_images_filepath"))
        self.training_labels_filepath = join(config("dataset_dir"), config("training_labels_filepath"))
        self.test_images_filepath = join(config("dataset_dir"), config("testing_images_filepath"))
        self.test_labels_filepath = join(config("dataset_dir"), config("testing_labels_filepath"))

    def _read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def _show_images(self, images, title_texts):
        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1
        for x in zip(images, title_texts):
            image = x[0]
            title_text = x[1]
            plt.subplot(rows, cols, index)
            plt.imshow(image, cmap=plt.cm.gray)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15)
            index += 1

    def load_data(self):
        x_train, y_train = self._read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self._read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

    def view(self, x_train, y_train, x_test, y_test):
        images = []
        titles = []
        for i in range(0, 10):
            r = random.randint(1, len(x_train))
            images.append(x_train[r])
            titles.append('training image [' + str(r) + '] = ' + str(y_train[r]))

        for i in range(0, 5):
            r = random.randint(1, len(x_test))
            images.append(x_test[r])
            titles.append('test image [' + str(r) + '] = ' + str(y_test[r]))

        self._show_images(images, titles)