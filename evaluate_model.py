# code from https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

import argparse
import glob
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from datetime import datetime

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-a", "--all", help="Whether or not all models should be evaluated.", action="store_true")
arg_parser.add_argument("-f", "--features", help="Path to features file.")
arg_parser.add_argument("-m", "--model", help="Path to model file.")
args = arg_parser.parse_args()

PATH_TO_FLICKR8K = '/home/caio/datasets/flickr8k/'
PATH_TO_FEATURES_FILE = args.features
PATH_TO_MODEL_FILE = args.model

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# def get_path_to_features_file(model_filename):
# 	for feature_filename in glob.glob("extracted_features/*.pkl"):
# 		if "effnetb" not in feature_filename:
# 			continue
# 		if find_effnet_version_in_filename(model_filename) == find_effnet_version_in_filename(feature_filename):
# 			return feature_filename
# 	raise Exception(f'Could not find feature file for the model {model_filename}')

# def find_effnet_version_in_filename(filename):
# 	return filename.split("effnetb")[1][0]

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	average_time_to_caption_an_image = 0
	for key, desc_list in descriptions.items():
		# generate description
		start_time = datetime.now()
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		end_time = datetime.now()
		average_time_to_caption_an_image += (end_time - start_time).total_seconds() * 1000
		# store actual and predicted
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	average_time_to_caption_an_image /= len(descriptions.items())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
	print(f'Average time taken to caption an image: {average_time_to_caption_an_image} ms')

# prepare tokenizer on train set

# load training dataset (6K)
filename = PATH_TO_FLICKR8K + 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = PATH_TO_FLICKR8K + 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

if args.all:
	model_filenames = glob.glob("saved_models/*.h5")
else:
	model_filenames = [PATH_TO_MODEL_FILE]
# load the model
for model_filename in model_filenames:
	# features_filename = get_path_to_features_file(model_filename)
	features_filename = PATH_TO_FEATURES_FILE
	test_features = load_photo_features(features_filename, test)
	print(f'--- Evaluating model {model_filename} with features from {features_filename}. ---')
	model = load_model(model_filename)
	# evaluate model
	evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)