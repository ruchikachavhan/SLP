import time 
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import math
import pickle
from data_loader import get_loader 
from vocab_build import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
from collections import namedtuple
import cv2
from trainer import Trainer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
	parser.add_argument('--crop_size', type=int, default=64 , help='size for randomly cropping images')
	parser.add_argument('--vocab_path', type=str, default='/home/ruchika/RSICD/vocab.pkl', help='path for vocabulary wrapper')
	parser.add_argument('--val_dir', type=str, default='/home/ruchika/RSICD/new/results', help='path for val')
	parser.add_argument('--image_dir', type=str, default='/home/ruchika/RSICD/RSICD_images/', help='directory for resized images')
	parser.add_argument('--caption_path', type=str, default='/home/ruchika/RSICD/annotations_RSICD/dataset_rsicd.json', help='path for train annotation json file')
	parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
	parser.add_argument('--max_seq', type=int , default=12, help='max seq length')
	parser.add_argument('--train_percent', type=float , default=0.95, help='training and validation')

	# Model parameters
	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=5000)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--learning_rate', type=float, default=0.0005)
	args = parser.parse_args()
	print(args)
	trainer = Trainer(args)
	trainer.train()
