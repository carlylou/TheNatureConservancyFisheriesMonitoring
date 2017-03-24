import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

import numpy as np
np.random.seed(2016)
target = 139
def l2_loss(y_true, y_pred):
	return K.sqrt(K.sum(K.square(y_pred - y_true), axis = -1))    
def load_from_file(filename):
	import numpy as np
	return np.load( filename + '.npy')
def get_shuffled():	
	coordinates = load_from_file('localization/localizer/train/shuffled_coordinates_float_139')
	resize_img = load_from_file('localization/localizer/train/shuffled_resize_img_float_139')
	Y = coordinates.reshape((-1,4))
	X = resize_img.reshape((-1,target,target,3))
	return X,Y

X,Y = get_shuffled()
# dimensions of our images.0
img_width, img_height = target, target
nb_epoch = 50

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (img_width,img_height,3))

x = base_model.output
x = AveragePooling2D()(x)
x = BatchNormalization(axis=1)(x)
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4)(x)
# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
	layer.trainable = False

sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(loss=l2_loss, optimizer=sgd)

train_data = X[660:]
train_labels = Y[660:]
validation_data = X[:660]
validation_labels = Y[:660]



ckpt = ModelCheckpoint('localization/localizer/inception/inception_localizer.h5', monitor='val_loss',
					verbose=0, save_best_only=True, save_weights_only=True)
callbacks = [
			EarlyStopping(monitor='val_loss', patience=5), ckpt
        ]
model.fit(train_data, train_labels,
			nb_epoch=nb_epoch, batch_size=32, shuffle=True,
			validation_data=(validation_data, validation_labels), callbacks=callbacks)
