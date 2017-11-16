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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '', "Data directory.")

#tf.python.control_flow_ops = tf

def fix_path(data_dir, img_filepath):
    return data_dir + "/IMG/" + img_filepath.split("/")[-1]

def preprocess_samples(samples):
    for line in samples:
        center_image = line[0]
        if (center_image == "center"):
            continue
        center_image = fix_path(FLAGS.data_dir, center_image)
        print('center image = {}'.format(center_image))
        image = cv2.imread(center_image)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        left_image = line[1]
        left_measurement = measurement + correction
        left_image_read = cv2.imread(left_image)
        images.append(left_image_read)
        measurements.append(left_measurement)

        right_image = line[2]
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
    return sklearn.utils.shuffle(X_train, Y_train)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(o, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        

lines = []
csv_file = FLAGS.data_dir + "/driving_log.csv"
print("Using CSV file {}".format(csv_file))
with open(csv_file) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

training_samples, validation_samples = train_test_split(lines, test_size = 0.2)

training_generator = generator(training_samples)
validation_generator = generator(validation_samples)

images = []
measurements = []

correction = 0.2
for line in lines:
    center_image = line[0]
    if (center_image == "center"):
        continue
    center_image = fix_path(FLAGS.data_dir, center_image)
    print('center image = {}'.format(center_image))
    image = cv2.imread(center_image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    left_image = line[1]
    left_measurement = measurement + correction
    left_image_read = cv2.imread(left_image)
    images.append(left_image_read)
    measurements.append(left_measurement)

    right_image = line[2]
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

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, Y_train, nb_epoch=5, validation_split=0.2, shuffle=True, verbose=1)
model.save('model.h5')

print(history.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

