import time 
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import math
import pickle
from torch.nn import Parameter
from data_loader import get_loader 
from vocab_build import Vocabulary
from models  import EncoderCNN, DecoderRNN
from modules import PolicyNet, ValueNet, LMLayer
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
from collections import namedtuple


use_cuda = 0
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {}

Transition = namedtuple('Transition', ('values'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def check_accuracy(predicted, original):
	acc = 0.00
	for i in range(0, len(predicted)):
		for j in range(0, original[0].shape[0]):
			if(predicted[i][0][0].item() == original[0][j].item()):
				acc+=1
	acc = acc/len(predicted)
	return acc

def hinge_loss(reward_diff, reward_sim):
	reward_diff = torch.sum(reward_diff)
	reward_sim = torch.sum(reward_sim)
	Loss = torch.zeros(1, 1)
	Loss[0] = max(0.00, 0.200 + reward_diff - reward_sim)
	return Loss

def get_vocab(vocab_path):
	f =  open(vocab_path, 'rb') 
	vocab = pickle.load(f)
	return vocab

class Trainer(object):
	def __init__(self, args):
		self.args = args
		self.vocab = get_vocab(args.vocab_path)
		self.Policynet = PolicyNet(args.embed_size, len(self.vocab), args.hidden_size, self.vocab, args.max_seq).to(device)
		self.valnet = ValueNet(args.embed_size, len(self.vocab), args.hidden_size, self.vocab, args.max_seq).to(device)
		self.lmlayer = LMLayer(args.embed_size, args.hidden_size).to(device)
		self.PolicynetParams = list(self.Policynet.CNNp.linear.parameters()) + list(self.Policynet.CNNp.norm1.parameters()) + list(self.Policynet.RNNp.parameters())
		self.ValnetParams = list(self.valnet.CNNv.linear.parameters()) + list(self.valnet.RNNv.parameters()) + list(self.valnet.fc1.parameters()) + list(self.valnet.fc2.parameters()) + list(self.valnet.fc3.parameters()) + list(self.valnet.norm1.parameters()) + list(self.valnet.norm2.parameters())
		self.LMlayerparans  =  list(self.lmlayer.fc1.parameters()) + list(self.lmlayer.fc2.parameters()) + list(self.lmlayer.fc3.parameters()) + list(self.lmlayer.norm1.parameters()) + list(self.lmlayer.norm2.parameters()) + list(self.lmlayer.norm3.parameters())
		self.policy_optimizer = torch.optim.Adam(self.PolicynetParams, lr = args.learning_rate)
		self.value_optimizer = torch.optim.Adam(self.ValnetParams, lr = args.learning_rate)
		self.lm_optimizer = torch.optim.Adam(self.LMlayerparans, lr = args.learning_rate)
		self.lr_scheduler_p = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size = 1, gamma = 0.9)
		self.lr_scheduler_v = torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size = 1, gamma = 0.9)
		self.lr_scheduler_e = torch.optim.lr_scheduler.StepLR(self.lm_optimizer, step_size = 1, gamma = 0.9)
		transform = transforms.Compose([transforms.RandomCrop(224),

										transforms.ToTensor(), 
										transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		self.train_data_loader, self.val_data_loader, self.val_names, self.train_names, self.airport, self.baseball, self.commercial, self.parking, self.stadium = get_loader(args.image_dir, args.caption_path, self.vocab, transform, args.batch_size, True, args.num_workers, args.train_percent, device)
		self.memory = ReplayMemory(10000)
		self.valnet.RNNv.load_state_dict(self.Policynet.RNNp.state_dict())
		self.valnet.RNNv.eval()
		self.valnet.CNNv.load_state_dict(self.Policynet.CNNp.state_dict())
		self.valnet.CNNv.eval()
	def train(self):
		vocab = self.vocab
		args = self.args	
		
		for epoch in range(0, args.num_epochs):
			print("-------------------------Epoch number-----------------------------", epoch+1)
			self.optimiseModels()
			self.validate(epoch)
			if(not(epoch == 0) and epoch % 2 == 0):
				self.lr_scheduler_p.step()
				self.lr_scheduler_v.step()
				self.lr_scheduler_e.step()
				# self.valnet.RNNv.load_state_dict(self.Policynet.RNNp.state_dict())
				# self.valnet.CNNv.load_state_dict(self.Policynet.CNNp.state_dict())

	def optimiseModels(self):
		args = self.args
		vocab = self.vocab
		index = 0
		GAMMA1 = 0.99
		GAMMA2 = 0.01
		for i, (images, captions, lengths) in enumerate(self.train_data_loader):
			# if(i<2):
			self.Policynet.zero_grad()
			self.valnet.zero_grad()
			self.lmlayer.zero_grad()
			print("************This is image number**************", i+1)
			print(self.train_names[i])

			images = Variable(images).to(device)
			captions = Variable(captions).to(device)
			print("Original captions", captions)
			targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
			features = self.valnet.features_extract(images)
			hiddens, hidden_list = self.valnet.captions_hidden_state(captions)
			Lmloss, reward = self.lmlayer.loss(features.detach(), hiddens.detach(), hidden_list)
			_ , captions_diff = self.choose_other_class(self.train_names[i])
			captions_diff = captions_diff.unsqueeze(0)
			captions_diff = captions_diff.long()
			captions_diff = Variable(captions_diff).to(device)
			hiddens_diff, hidden_list_diff = self.valnet.captions_hidden_state(captions_diff)
			Lmloss_diff, reward_diff = self.lmlayer.loss(features.detach(), hiddens_diff.detach(), hidden_list_diff)
			print("Rewards for similar images", torch.mean(reward).item())
			print("Rewards for dissimilar images", torch.mean(reward_diff).item())
			LossLm = torch.sum(Lmloss) + torch.sum(Lmloss_diff) +  hinge_loss(reward_diff, reward)
			print("Linear mapping layer loss", LossLm.item())
			LossLm.backward()
			self.lm_optimizer.step()
			policyloss, generated_embed = self.Policynet.loss(images, captions)
			policyloss.backward(retain_graph = True)
			print("Policy net loss", policyloss.item())
			self.policy_optimizer.step()
			valLoss = self.valnet.loss(images, reward.detach(), generated_embed)
			print("Value net loss", valLoss.item())
			valLoss.backward()
			self.value_optimizer.step()
		
			
	def validate(self, epoch):
		args = self.args
		vocab = self.vocab
		print("-----------------Validation going on------------------------")
		for v, (images_val, captions_val, lengths_val) in enumerate(self.val_data_loader):
			# with torch.no_grad():	
			images_val = Variable(images_val).to(device)
			captions_val = Variable(captions_val).to(device)
			Q_vals, _ , predicted = self.Policynet.forward(images_val, captions_val)
			sentence, ids = self.Policynet.RNNp.get_sentences_test(predicted, vocab)
			print(sentence)
			print("Validation Accuracy", check_accuracy(ids, captions_val))
			if(epoch == 0):
				name = "results/" + self.val_names[v]+ '.txt'
				f = open(name, "w")
				f.write(str(epoch))
				f.write("\n")
				f.write(str(sentence))
				f.write("\n")
			else:
				name = "results/" + self.val_names[v]+ '.txt'
				f = open(name, "a+")
				f.write(str(epoch))
				f.write("\n")
				f.write(str(sentence))
				f.write("\n")

	def choose_other_class(self, name): 
		name = name.split("_")[0]
		if(name == 'airport'):
			index = np.random.randint(0, 2)
			if(index == 0):
				id = np.random.randint(0, len(self.baseball))
				return self.baseball.dataset[id]
			# if(index == 1):
			# 	id = np.random.randint(0, len(self.commercial))
			# 	return self.commercial.dataset[id]
			if(index == 1):
				id = np.random.randint(0, len(self.parking))
				return self.parking.dataset[id]
			# if(index == 3):
			# 	id = np.random.randint(0, len(self.stadium))
			# 	return self.stadium.dataset[id]
		if(name == 'baseballfield'):
			index = np.random.randint(0, 2)
			if(index == 0):
				id = np.random.randint(0, len(self.airport))
				return self.airport.dataset[id]
			# if(index == 1):
			# 	id = np.random.randint(0, len(self.commercial))
			# 	return self.commercial.dataset[id]
			if(index == 1):
				id = np.random.randint(0, len(self.parking))
				return self.parking.dataset[id]
			# if(index == 3):
			# 	id = np.random.randint(0, len(self.stadium))
			# 	return self.stadium.dataset[id]
		if(name == 'parking'):
			index = np.random.randint(0, 2)
			if(index == 0):
				id = np.random.randint(0, len(self.baseball))
				return self.baseball.dataset[id]
			# if(index == 1):
			# 	id = np.random.randint(0, len(self.commercial))
			# 	return self.commercial.dataset[id]
			if(index == 1):
				id = np.random.randint(0, len(self.airport))
				return self.airport.dataset[id]
			# if(index == 3):
			# 	id = np.random.randint(0, len(self.stadium))
			# 	return self.stadium.dataset[id]
		# if(name == 'stadium'):
		# 	index = np.random.randint(0, 4)
		# 	if(index == 0):
		# 		id = np.random.randint(0, len(self.baseball))
		# 		return self.baseball.dataset[id]
		# 	if(index == 1):
		# 		id = np.random.randint(0, len(self.commercial))
		# 		return self.commercial.dataset[id]
		# 	if(index == 2):
		# 		id = np.random.randint(0, len(self.parking))
		# 		return self.parking.dataset[id]
		# 	if(index == 3):
		# 		id = np.random.randint(0, len(self.airport))
		# 		return self.airport.dataset[id]
		# if(name == 'commercial'):
		# 	index = np.random.randint(0, 4)
		# 	if(index == 0):
		# 		id = np.random.randint(0, len(self.baseball))
		# 		return self.baseball.dataset[id]
		# 	if(index == 1):
		# 		id = np.random.randint(0, len(self.stadium))
		# 		return self.stadium.dataset[id]
		# 	if(index == 2):
		# 		id = np.random.randint(0, len(self.parking))
		# 		return self.parking.dataset[id]
		# 	if(index == 3):
		# 		id = np.random.randint(0, len(self.airport))
		# 		return self.airport.dataset[id]


