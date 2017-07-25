import csv
import cv2
import numpy as np

lines = []
with open("./driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]

    filename_center = "./IMG/" + source_path_center.split('\\')[-1]
    filename_left = "./IMG/" + source_path_left.split('\\')[-1]
    filename_right = "./IMG/" + source_path_right.split('\\')[-1]

    image_center = cv2.imread(filename_center)
    image_left = cv2.imread(filename_left)
    image_right = cv2.imread(filename_right)

    images.append(image_center)
    ##images.append(image_left)
    ##images.append(image_right)

    measurement_center = float(line[3])
    measurement_left= float(line[3]) + 0.25
    measurement_right = float(line[3]) - 0.25

    measurements.append(measurement_center)
    ##measurements.append(measurement_left)
    ##measurements.append(measurement_right)

    image_flipped = np.fliplr(image_center)
    images.append(image_flipped)
    measurement_flipped = -measurement_center
    measurements.append(measurement_flipped)

X_train = np.array(images)
Y_train = np.array(measurements)

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

model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')
