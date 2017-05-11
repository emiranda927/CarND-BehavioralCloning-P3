import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

#initialize arrays for camera images from 3 cameras
images = []
img_center = []
img_left = []
img_right = []
measurements = []

#define data paths
data_paths = ['./data/TestData1-Normal/',\
              './data/TestData2-Reverse/',\
              './data/TestData4-Normal2/',\
              './data/TestData5-Reverse2/',\
              './data/TestData6-Recovery/',\
              './data/Provided-Data/'
             ]

#load path information for each folder
samples = []
for path in data_paths:
    lines = []
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    samples.extend(lines)

#create generator to provide batches of images to CNN            
def generator(samples, batch_size=64):
    num_samples = len(samples)
    correction = 0.25
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            #load each image -- apply steering correction to L/R camera images
            for line in batch_samples:
                steering_angle = float(line[3])
                steering_left = steering_angle+correction
                steering_right = steering_angle-correction
                img_center = cv2.imread(line[0].strip())
                img_left = cv2.imread(line[1].strip())
                img_right = cv2.imread(line[2].strip())
        
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_angle, steering_left, steering_right])

            #augment data by horizontally flipping images
            flipped_images, flipped_angles = [], []
            for image, angle in zip(images, angles):
                flipped_images.append(image)
                flipped_angles.append(angle)
                flipped_angles.append(angle*-1.0)
                flipped_images.append(cv2.flip(image,1))

            #yield augmented batch of images    
            X_train = np.array(flipped_images)
            y_train = np.array(flipped_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#split training/validation sets - create generators for each
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#define CNN architecture - NVIDIA End-to-End DNN with LeakyReLU activations
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, GaussianDropout
from keras.layers.advanced_activations import LeakyReLU
model = Sequential()
model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5)) #Normalize images
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2)))
model.add(LeakyReLU(alpha=0.3)) #LeakyReLU appeared to make the vehicle drive more smoothly
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2)))
model.add(LeakyReLU(alpha=0.3))
model.add(GaussianDropout(rate=0.5)) #Gaussian Dropout to generalize and prevent overfitting
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2)))
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(rate=0.5)) #Dropout
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(LeakyReLU(alpha=0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(rate=0.5)) #Dropout
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse') #used adam optimizer for parameter tuning

#fit model
history = model.fit_generator(train_generator, steps_per_epoch= \
            len(train_samples)/64, validation_data=validation_generator, \
            validation_steps=len(validation_samples)/ 64, epochs=10)

model.save('model.h5') #save model

#Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()
