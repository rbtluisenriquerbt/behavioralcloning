import csv
import cv2
import numpy as np

samples = []
with open("./driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

images = []
measurements = []

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename_center = './IMG/'+batch_sample[0].split('\\')[-1]
                filename_left = './IMG/'+batch_sample[1].split('\\')[-1]
                filename_right = './IMG/'+batch_sample[2].split('\\')[-1]

                image_center = cv2.imread(filename_center)
                image_left = cv2.imread(filename_left)
                image_right = cv2.imread(filename_right)

                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + 0.25
                right_angle = float(batch_sample[3]) - 0.25
                images.append(image_center)
                images.append(image_left)
                images.append(image_right)

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# for line in train_samples:
#     source_path_center = line[0]
#     source_path_left = line[1]
#     source_path_right = line[2]
#
#     filename_center = "./IMG/" + source_path_center.split('\\')[-1]
#     filename_left = "./IMG/" + source_path_left.split('\\')[-1]
#     filename_right = "./IMG/" + source_path_right.split('\\')[-1]
#
#     image_center = cv2.imread(filename_center)
#     image_left = cv2.imread(filename_left)
#     image_right = cv2.imread(filename_right)
#
#     images.append(image_center)
#     ##images.append(image_left)
#     ##images.append(image_right)
#
#     measurement_center = float(line[3])
#     measurement_left= float(line[3]) + 0.25
#     measurement_right = float(line[3]) - 0.25
#
#     measurements.append(measurement_center)
#     ##measurements.append(measurement_left)
#     ##measurements.append(measurement_right)
#
#     image_flipped = np.fliplr(image_center)
#     images.append(image_flipped)
#     measurement_flipped = -measurement_center
#     measurements.append(measurement_flipped)

##X_train = np.array(images)
##Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: ((x / 255.0) - 0.5)))

model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
