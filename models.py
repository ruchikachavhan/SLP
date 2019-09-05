import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import math
import numpy as np
import random

use_cuda = 0
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {}

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)#USing pretrained Alexnet
        in_features = self.alexnet.classifier[6].in_features
        self.linear = nn.Linear(in_features, embed_size)
        self.alexnet.classifier[6] = self.linear
        self.norm1 = nn.LayerNorm(embed_size)
    def forward(self, images):
        """Extract feature vectors from input images.""" 
        features = F.relu(self.norm1(self.alexnet(images))) #Will only update linear layer parameters
        return features

class DecoderRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, vocab, max_seq):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)
		self.linear = nn.utils.weight_norm(nn.Linear(hidden_dim, vocab_size))
		self.linear1 = nn.utils.weight_norm(nn.Linear(vocab_size, vocab_size))
		self.relu = nn.LeakyReLU(0.3, inplace = True)
		self.vocab = vocab
		self.layernorm = nn.LayerNorm(vocab_size)
		self.layernorm1 = nn.LayerNorm(vocab_size)
		self.layernormc = nn.LayerNorm(vocab_size)
		self.layernormc1 = nn.LayerNorm(vocab_size)
		self.max_seq = max_seq
		self.hidden_dim = hidden_dim
		self.embed_dim = embedding_dim
		self.EPS_START = 10.2
		self.EPS_END = 0.05
		self.EPS_DECAY = 1000
		self.steps_done = 0
		self.words = []
	def forward(self, features, captions):
		self.words.clear()
		self.max_seq = captions.shape[1]
		features = features.unsqueeze(1) #to make a 3D tensor
		bs = features.shape[0] #features shape is (bs, 1, 256)
		Q_vals = torch.zeros(bs, self.max_seq, 1, 1, len(self.vocab))
		hid_state_t = torch.zeros(bs, self.max_seq,1, 1, self.hidden_dim) # for lstm output
		generated_embed = torch.zeros(bs, self.max_seq, 1, 1, self.embed_dim)
		predicted_list = torch.zeros(bs, self.max_seq, 1)
		self.steps_done += 1
		for b in range(0, bs):
			Q_vals_bs = torch.zeros(self.max_seq, 1 , 1, len(self.vocab))
			hid_state_t_bs = torch.zeros(self.max_seq, 1 , 1, self.hidden_dim)
			pred_bs = torch.zeros(self.max_seq, 1 )
			generated_embedbs = torch.zeros(self.max_seq, 1, 1, self.embed_dim)
			hiddens = None
			in_features = features[b].unsqueeze(0) # to make a 3d tensor
			for m in range(self.max_seq):
				lstm_out, hiddens = self.lstm(in_features, hiddens) # size(1, 1, 512), 
				#Hiddens is a tuple of (h_t, c_t) and we need to use h_t, h_t size = lstm_out size
				hid_state_t_bs[m] = lstm_out
				linear_layer = self.relu(self.layernorm(self.linear(lstm_out)))
				linear_layer1 = self.layernorm1(self.linear1(linear_layer))
				out = F.softmax(linear_layer1, dim = 2) #size(1, 1, len(vocab)), need to add dimension of 2 in softmax otherwise output is all one because it processes 1, 1
				Q_vals_bs[m] = out
				predicted = self.select_action(out).to(device)
				pred_bs[m] = predicted
				embed = self.get_embedding(predicted)
				word = self.get_word(predicted)
				self.words.append(word)
				generated_embedbs[m] =  embed
				in_features = embed #Loop in lSTM
			Q_vals[b] = Q_vals_bs
			hid_state_t[b] = hid_state_t_bs
			generated_embed[b] = generated_embedbs
			predicted_list[b] = pred_bs
		return Q_vals, hid_state_t, generated_embed, predicted_list

	def get_word(self, predicted):
		word_id = predicted.item()
		word = self.vocab.idx2word[word_id]
		return word

	def get_embedding(self, word):
		embed = self.embed(word)
		return embed

	def print_worss(self):
		print(self.words)

	def forward_captions(self, captions):
		self.words.clear()
		max_seq = captions.shape[1]
		hiddens = None
		hidden_list = []
		self.steps_done += 1
		for b in range(0, captions.shape[1]):
			hidden_listb= torch.zeros(max_seq, 1, 1, self.hidden_dim)
			in_features = captions[:, b].unsqueeze(0) # to make a 3d tensor
			for m in range(0, max_seq):
				lstm_out, hiddens = self.lstm(in_features, hiddens) 
				# size(1, 1, 512), #Hiddens is a tuple of (h_t, c_t) and we need to use h_t, h_t size = lstm_out size
				linear_layer = self.relu(self.layernormc(self.linear(lstm_out)))
				linear_layer1 = self.layernormc1(self.linear1(linear_layer))
				out = F.softmax(linear_layer1, dim = 2)
				predicted = self.select_action(out).to(device)				
				embed = self.get_embedding(predicted)
				in_features = embed
				hidden_listb[m] = hiddens[0].to(device)
			hidden_list.append(hidden_listb)
		return hiddens[0], hidden_list

	def get_sentences_test(self, rnn_output, vocab):
		sampled_caption = []
		sampled_id = []
		sentences = []
		for index in range(0, rnn_output.shape[0]):
			sampled_caption = []
			for i in range(0, rnn_output[index].shape[0]):
				predicted = rnn_output[index][i]
				# predicted = select_action(rnn_output[index][i])
				sampled_ids = []                                 
				sampled_ids.append(predicted)
				sampled_id.append(sampled_ids)
				for word_id in sampled_ids:
					word_id = word_id.item()
					word = vocab.idx2word[word_id]
					sampled_caption.append(word)
					if word == '<end>':
						break
			sentence = ' '.join(sampled_caption)
			sentences.append(sentence)
		return sentences, sampled_id

	def select_action(self, state):
		sample = random.random()
		eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
		math.exp(-1. * self.steps_done / self.EPS_DECAY)
		if sample > eps_threshold:
			_, predicted = state.max(2)
			return predicted
		else:
			index =  torch.randint(0, len(self.vocab), (1, 1))
			return index