import cv2
import numpy as np
from glob import glob
import os

from keras import losses, optimizers, regularizers
from keras.models import Sequential
from keras.layers import Flatten, Lambda,  Dense, Dropout, Activation
from keras.layers import Dropout, Conv2D, Convolution2D, MaxPooling2D, Cropping2D
from keras.utils.np_utils import to_categorical

################################################################################
#           python generator logic to load images on the fly + augmentation logic
################################################################################

# this is in the order as it is in TrafficLight.msg
categories = ['red', 'yellow', 'green', 'none']
categories_label = [ [1,0,0] , [0,1,0], [0,0,1], [0,0,0]]

images = []
labels = []

for img_class, directory in enumerate(categories):
    for file_name in glob("tl_images/{}/*.jpg".format(directory)):
        file = cv2.imread(file_name)
        file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR);
        resized = cv2.resize(file, (32,32))

        images.append(resized)
        labels.append(categories_label[img_class])

images = np.array(images)
labels = np.array(labels)

################################################################################
#       normalization
################################################################################

model = Sequential()
model.add( Lambda(lambda x: x/255.0, input_shape=(32,32,3)))
model.add( Conv2D( 6, (5, 5), padding='same', input_shape=(32,32,3), activation='relu') )
model.add( MaxPooling2D() )
model.add( Conv2D( 6, (5, 5), padding='same',  activation='relu') )
model.add( MaxPooling2D() )
model.add( Flatten())
model.add( Dense(120) )
model.add( Dense(60) )
model.add( Activation("relu") )
#softmax classifier
model.add(Dense(3))
model.add(Activation("softmax"))


################################################################################
#       keras run
################################################################################

model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])

model.fit(images, labels, epochs=50, verbose=True, validation_split=0.3, shuffle=True)

model.save('tl_classifier.h5')