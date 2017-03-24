#!/usr/bin/env python
# encoding: UTF-8

# import the necessary packages
import imutils
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
import numpy as np
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid

	while True:
		# compute the new dimensions of the image and resize it 缩小图片尺寸
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image

def save_to_file(filename, target):
	import numpy as np
	np.save(filename, target)

def load_from_file(filename):
	import numpy as np
	return np.load( filename + '.npy')

def load_all_fishdata():
    train_data = load_from_file('load/train_data')
    trainY_onehot = load_from_file('load/trainY_onehot')
    train_id = load_from_file('load/train_id')
    test_data = load_from_file('load/test_data')
    test_id = load_from_file('load/test_id')
    return train_data, trainY_onehot, train_id, test_data, test_id

def get_im_cv2(path):
	import cv2
	img = cv2.imread(path)
	resized = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
	return resized

def load_test(foldname):
	import time
	path = os.path.join('..', 'input', 'test_stg1', foldname,'*.jpg')
	files = sorted(glob.glob(path))
	X_test = []
	X_test_id = []
	for fl in files:
		flbase = os.path.basename(fl)
		img = get_im_cv2(fl)
		X_test.append(img)
		X_test_id.append(flbase)
	return X_test, X_test_id

def read_and_normalize_test_data(name):
	import time
	start_time = time.time()
	test_data, test_id = load_test(name)

	test_data = np.array(test_data, dtype=np.uint8)
	test_data = test_data.transpose((0, 3, 1, 2))

	test_data = test_data.astype('float32')
	test_data = test_data / 255

	print('Test shape:', test_data.shape)
	print(test_data.shape[0], 'test samples')
	print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
	return test_data, test_id

def merge_several_folds_mean(data, nfolds):
	a = np.array(data[0])
	for i in range(1, nfolds):
		a += np.array(data[i])
	a /= nfolds
	return a.tolist()

def create_submission(predictions, test_id, info):
	result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
	result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
	now = datetime.datetime.now()
	sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
	result1.to_csv(sub_file, index=False)












