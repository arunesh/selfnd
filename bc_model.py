# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
import csv
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', "Data directory.")

#tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def fix_path(data_dir, img_filepath):
    return data_dir + "/IMG/" + img_filepath.split("/")[-1]

correction = 0.2
def preprocess_samples(samples):
    images = []
    measurements = []
    for line in samples:
        center_image = line[0]
        if (center_image == "center"):
            continue
        center_image = fix_path(FLAGS.data_dir, center_image)
        image = cv2.imread(center_image)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        left_image = fix_path(FLAGS.data_dir, line[1])
        left_measurement = measurement + correction
        left_image_read = cv2.imread(left_image)
        images.append(left_image_read)
        measurements.append(left_measurement)

        right_image = fix_path(FLAGS.data_dir, line[2])
        right_measurement = measurement - correction
        right_image_read = cv2.imread(right_image)
        images.append(right_image_read)
        measurements.append(right_measurement)
        
    aug_measurements = []
    aug_images = []

    for image, measurement in zip(images, measurements):
        aug_images.append(image)
        aug_measurements.append(measurement)
        aug_images.append(cv2.flip(image, 1))
        aug_measurements.append(measurement * -1.0)

    X_train = np.array(aug_images)
    Y_train = np.array(aug_measurements)
    return shuffle(X_train, Y_train)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            yield preprocess_samples(batch_samples)
        
lines = []
csv_file = FLAGS.data_dir + "/driving_log.csv"
print("Using CSV file {}".format(csv_file))
with open(csv_file) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

training_samples, validation_samples = train_test_split(lines, test_size = 0.2)

print("Training set size = {}".format(len(training_samples)))
print("Validation set size = {}".format(len(validation_samples)))

training_generator = generator(training_samples)
validation_generator = generator(validation_samples)


ch, row, col = 3, 90, 320
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch),
                output_shape=(row, col, ch)))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(training_generator, samples_per_epoch = len(training_samples), validation_data = validation_generator, \
        nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')

print(history.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
