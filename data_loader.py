import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import json
from vocab_build import Vocabulary
import argparse
from models  import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from modules import PolicyNet, ValueNet
# To shuffle names list and image+captions list in unison
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	A = []
	B = []
	p = np.random.permutation(len(a))
	for i in range(0, len(a)):
		A.append(a[p[i]])
		B.append(b[p[i]])
	return A, B

def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).

	We should build custom collate_fn rather than using default collate_fn, 
	because merging caption (including padding) is not supported in default.

	Args:
	data: list of tuple (image, caption). 
	- image: torch tensor of shape (3, 224, 224).
	- caption: torch tensor of shape (?); variable length.

	Returns:
	images: torch tensor of shape (batch_size, 3, 224, 224).
	targets: torch tensor of shape (batch_size, padded_length).
	lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)
	# print("in collate_fn", images.shape)
	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]
	return images, targets, lengths


def get_loader(root, json_file, vocab, transform, batch_size, shuffle, num_workers, train_percent, device):
	data = json.loads(open(json_file, "r").read())
	airport = []
	baseball = []
	commercial = []
	parking = []
	stadium = [] 
	imgCaptionList = []
	images_names_list = []
	for index in range(0, len(data['images'])):
		# if(index<30):
		print("Loading Dataset image number", index+1)
		image = data['images'][index]['filename']
		print(image)
		img = Image.open(os.path.join(root, image)).convert('RGB')
		img_t = transform(img)
		img.close()
		sentence = data['images'][index]['sentences'][3]['raw'] 
		tokens = nltk.tokenize.word_tokenize(str(sentence))
		caption = []
		caption.append(vocab('<start>'))
		caption.extend([vocab(token) for token in tokens])
		caption.append(vocab('<end>'))
		target = torch.Tensor(caption)
		data_point = (img_t, target)
		#Appending in different list according to class
		if(image.split("_")[0] == 'airport'):
			airport.append(data_point)
			imgCaptionList.append(data_point)
			images_names_list.append(image)
		if(image.split("_")[0] == 'baseballfield'):
			baseball.append(data_point)
			imgCaptionList.append(data_point)
			images_names_list.append(image)
		# if(image.split("_")[0] == 'commercial'):
		# 	commercial.append(data_point)
		# 	imgCaptionList.append(data_point)
		# 	images_names_list.append(image)
		if(image.split("_")[0] == 'parking'):
			parking.append(data_point)
			imgCaptionList.append(data_point)
			images_names_list.append(image)
		# if(image.split("_")[0] == 'stadium'):
		# 	stadium.append(data_point)
		# 	imgCaptionList.append(data_point)
		# 	images_names_list.append(image)
		#Appending in one list for train and val dataloader
		del caption, img_t, data_point, target

	imgCaptionList, images_names_list = unison_shuffled_copies(imgCaptionList, images_names_list) # shuffling the dataset with names and images and captions
	train_data = imgCaptionList[0: int(len(imgCaptionList)*train_percent)]
	train_names = images_names_list[0: int(len(imgCaptionList)*train_percent)]
	val_data = imgCaptionList[int(len(imgCaptionList)*train_percent): len(imgCaptionList)]
	val_names = images_names_list[int(len(images_names_list)*train_percent): len(images_names_list)]

	train_data_loader = torch.utils.data.DataLoader(dataset=train_data, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	airport = torch.utils.data.DataLoader(dataset=airport, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	baseball = torch.utils.data.DataLoader(dataset=baseball, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	commercial = torch.utils.data.DataLoader(dataset=commercial, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	stadium = torch.utils.data.DataLoader(dataset=stadium, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	parking = torch.utils.data.DataLoader(dataset=parking, 
											batch_size=batch_size,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)


	val_data_loader = torch.utils.data.DataLoader(dataset=val_data, 
											batch_size=1,
											shuffle=False,
											num_workers=4,
											collate_fn=collate_fn)
	print("-------------------------------------DATASET LOADING DONE---------------------------------------")
	print("Number of training samples", len(train_data))
	print("Number of validation samples", len(val_data))
	print("Images in validation set", val_names)
	print("Number of airport images", len(airport))
	print("Number of baseball images", len(baseball))
	print("Number of comm images", len(commercial))
	print("Number of parking images", len(parking))
	del imgCaptionList, images_names_list, train_data, val_data
	return train_data_loader, val_data_loader, val_names, train_names, airport, baseball, commercial, parking, stadium


