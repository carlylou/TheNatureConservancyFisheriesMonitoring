from skimage.transform import AffineTransform, matrix_transform, warp, rotate
import glob
import os
import numpy as np
import pandas as pd

def get_coor_image_train():
	print 'Start detail with images:'
	folders = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
	jsons = ['alb_labels.json', 'bet_labels.json', 'dol_labels.json', 'lag_labels.json', 'other_labels.json', 'shark_labels.json', 'yft_labels.json']
	coordinates = np.array([])
	resize_img = []
	resize_id = []
	for fld in folders:
		index = folders.index(fld)
		json_name = jsons[index]
		json_dir = '../kaggleNatureConservancy-master/'+json_name
		fish_labels = pd.read_json(json_dir)
		print('process folder {} (Index: {})'.format(fld, index))
		path = os.path.join('..','input','train', fld, '*.jpg')
		files = glob.glob(path)
		for fl in files:
			flbase = os.path.basename(fl)
			#print flbase
			json_index = fish_labels[fish_labels.filename == flbase].index.values[0]
			resized, coordinate, dst, coord2 = get_im_coor(fl, fish_labels.iloc[json_index])
			if(resized == None and coordinate == None ):
				continue
			coordinates = np.concatenate((coordinates, coordinate), axis = 0)
			coordinates = np.concatenate((coordinates, coord2), axis = 0)
			resize_img.append(resized)
			resize_img.append(dst)
			resize_id.append(flbase)
			resize_id.append(flbase)
		print fld
	# save the result,图像id不能转成数组，不然就不是原来的数据了，:
	resize_img = np.array(resize_img, dtype=np.uint8)
	resize_img = resize_img.astype('float32')
	import helpers
	helpers.save_to_file('localization/localizer/train/coordinates_float_139',coordinates)
	helpers.save_to_file('localization/localizer/train/resize_img_float_139',resize_img)
	helpers.save_to_file('localization/localizer/train/resize_id_float_139',resize_id)
	print 'success'


def get_im_coor(path, json):
	import math
	import cv2
	import numpy as np
	img = cv2.imread(path)
	rows = img.shape[0]# height y
	columns = img.shape[1]# width x 
	# get the head and tail coordinates
	if(np.size(json.annotations) == 0):
		return None, None,None, None
	x1 = json.annotations[0][u'x']
	x2 = json.annotations[1][u'x']
	y1 = json.annotations[0][u'y']
	y2 = json.annotations[1][u'y']
	# find appropriate placement
	x1_ = np.maximum(0, np.minimum(x1,x2)-50) 
	x2_ = np.minimum(columns, np.maximum(x1,x2)+50)
	y1_ = np.maximum(0, np.minimum(y1,y2)-50)
	y2_ = np.minimum(rows, np.maximum(y1,y2)+50)
	target = 139
	resized = cv2.resize(img, (target, target), cv2.INTER_LINEAR)
	height, width = img.shape[:2]
	x_ratio = float(target)/float(width)
	y_ratio = float(target)/float(height)
	x1_r = x1_*x_ratio
	y1_r = y1_*y_ratio
	x2_r = x2_*x_ratio
	y2_r = y2_*y_ratio
	# data augmentation:
	M = np.array([[  0.85,  -0.15,  18.  ],[ -0.15,   0.85,  18.  ]])
	M1 = np.concatenate((M, np.array([[0,0,1]])),axis = 0)
	# get the transformed image
	dst = cv2.warpAffine(resized, M, (target, target))
	coord = np.array([[x1_r,y1_r],[x2_r,y2_r]])
	coord1 = matrix_transform(coord, M1)
	# resized x0, y0, w, h
	resized_array = np.array([x1_r, y1_r, x2_r-x1_r, y2_r-y1_r])
	dst_array = np.array([coord1[0][0], coord1[0][1], coord1[1][0]-coord1[0][0], coord1[1][1]-coord1[0][1]])
	return resized, resized_array, dst, dst_array

get_coor_image_train()




