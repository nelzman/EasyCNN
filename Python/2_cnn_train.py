# -*- coding: utf-8 -*-
# Part 1 - Building the CNN

# set Workingdirectory

name = 'Felix Fritzsche'

kategorien = ['Car', 'Laptop', 'Smartphone']

picture_limit = 20

#No GPU.
# before importing tensorflow backend
#from __future__ import print_function
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Part 1 - Building the CNN

# set Workingdirectory
import os
#os.chdir(r'Path\to\your\pictures')

import random
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop

from google_images_download import google_images_download   #importing the library
import shutil
import numpy as np

# Pretrained Convolutional Layers

conv_base_vgg16 = VGG16(
  weights = "imagenet",
  include_top = False,
  input_shape = (150, 150, 3)
)

conv_base_vgg16.trainable = False

conv_base_vgg16.summary()

# Initialising the CNN
classifier = Sequential()

classifier.add(conv_base_vgg16)

classifier.add(Flatten(input_shape=conv_base_vgg16.output_shape[1:]))
# =============================================================================
# 
# # Step 1 - Convolution
# classifier.add(Convolution2D(128, kernel_size = (3,3), strides = (3, 3), input_shape = (150,150, 3), activation = 'relu'))
# 
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Adding a second convolutional layer
# classifier.add(Convolution2D(64, kernel_size = (3,3), strides = (3, 3), activation = 'relu'))
# 
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Step 3 - Flattening
# classifier.add(Flatten())
# 
# =============================================================================
# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.compile(optimizer = RMSprop(lr = 2e-5), loss ='categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255,
                                     rotation_range = 40,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     shear_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip = True,
                                     fill_mode = "nearest")

test_datagen = ImageDataGenerator(rescale = 1./255)

#training_set_floor = train_datagen.flow_from_directory('bilder/training_set',
#                                                 batch_size = 5,
#                                                 class_mode = 'binary')

#classifier.fit_generator(training_set_floor,
#                         samples_per_epoch = 5,
#                         nb_epoch = 10)

training_set = train_datagen.flow_from_directory('pics/dataset/training_set',
                                                 target_size = [150,150],
                                                 batch_size = 5, 
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('pics/dataset/test_set',
                                            target_size = [150,150],
                                            batch_size = 5, 
                                            class_mode = 'categorical')


# Checkpoints des CNN(nach jeder Epoch wird gespeichert):
checkpoint_dir = "checkpoints"
#os.makedirs(checkpoint_dir)
filepath = checkpoint_dir + "/model.{epoch:02d}-{loss:.3f}-{acc:.2f}-{val_acc:.2f}.hdf5"

# Create checkpoint callback
cp_callback = ModelCheckpoint(
  filepath = filepath,
  save_weights_only = False,
  verbose = 1
)
###############################################################################

classifier.fit_generator(training_set,
                         samples_per_epoch = 200,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 20,
                         callbacks = [cp_callback])

# Save Data

classifier.save("cnn_model_dogcats.h5")
