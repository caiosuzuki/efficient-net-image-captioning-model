# code from https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

import argparse
from os import listdir
from pickle import dump
# from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from efficientnet import *


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-f", "--filepath", help="Filepath to store the desired features file (.pkl).")
arg_parser.add_argument("-v", "--version", help="Effnet's version to be used.")
args = arg_parser.parse_args()

features_file_path = args.filepath
effnet_version = args.version

configurations_for_effnet_version = {
	"effnetb0": {
		"img_size": 224,
		"model_function": EfficientNetB0()
		},
	"effnetb1": {
		"img_size": 240,
		"model_function": EfficientNetB1()
		},
	"effnetb2": {
		"img_size": 260,
		"model_function": EfficientNetB2()
		},
	"effnetb3": {
		"img_size": 300,
		"model_function": EfficientNetB3()
		},
	"effnetb4": {
		"img_size": 380,
		"model_function": EfficientNetB4()
		},
	"effnetb5": {
		"img_size": 456,
		"model_function": EfficientNetB5()
		},
	"effnetb6": {
		"img_size": 528,
		"model_function": EfficientNetB6()
		},
	"effnetb7": {
		"img_size": 600,
		"model_function": EfficientNetB7()
		}
}

# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	# model = VGG16()
	model = configurations_for_effnet_version[effnet_version]['model_function']
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		img_size = configurations_for_effnet_version[effnet_version]['img_size']
		image = load_img(filename, target_size=(img_size, img_size))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the model
		# ATTENTION: in order to use VGG16 or ResNet, you should use their specific preprocess_input functions
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = '/home/caio/datasets/flickr8k/Flickr8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open(features_file_path, 'wb'))