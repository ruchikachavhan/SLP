import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from models  import EncoderCNN, DecoderRNN
import numpy as np

use_cuda = 0
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {}

class PolicyNet(nn.Module):
	def __init__(self, embed_size, vocab_size, hidden_size, vocab, max_seq):
		super(PolicyNet, self).__init__()
		self.embed_size = embed_size
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.vocab = vocab
		self.CNNp = EncoderCNN(embed_size)
		self.RNNp = DecoderRNN(vocab_size, embed_size, hidden_size, vocab, max_seq)
	def forward(self, images, captions):
		features = self.CNNp(images)
		Q_vals, _, generated_embed, predicted= self.RNNp(features, captions)
		return Q_vals, generated_embed , predicted
	def loss(self, image, captions):
		criterion = nn.NLLLoss()
		Q_vals , generated_embed, predicted = self.forward(image, captions)
		predicted = predicted.view(predicted.shape[1])
		Q_vals = Q_vals.squeeze(2)
		Q_vals = Q_vals.squeeze(2)
		loss = torch.zeros(1, 1)
		for i in range(0, Q_vals.shape[0]):
			for j in range(0, Q_vals.shape[1]):
				q = Q_vals[i][j][int(predicted[j].item())]
				loss[i] = loss[i] - torch.log(q + 10e-16)
		print("Length of sentence", len( self.RNNp.words))
		print("words while training", self.RNNp.words)
		return loss, generated_embed


class LMLayer(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(LMLayer, self).__init__()
		self.fc1 = nn.utils.weight_norm(nn.Linear(embed_size, hidden_size))
		self.fc2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
		self.fc3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
		self.alpha = 0.2
		self.hidden_dim = hidden_size
		self.norm1 = nn.LayerNorm(hidden_size)
		self.norm2 = nn.LayerNorm(hidden_size)
		self.norm3 = nn.LayerNorm(hidden_size)
	def forward(self, features):
		fc_1 = F.relu(self.norm1(self.fc1((features))))
		fc_1 = F.relu(self.norm2(self.fc2(fc_1)))
		fc_1 = F.relu(self.norm3(self.fc3(fc_1)))
		return fc_1

	def loss(self, features, hidden_states, hidden_list):
		#Calculating reward
		eps = 10e-12
		fc_1 = self.forward(features)
		fc_1 = fc_1.squeeze(0)
		fc_1 = fc_1.to(device)
		hidden_states = hidden_states.squeeze(0).view(hidden_states.shape[2], 1)
		
		reward = torch.zeros(len(hidden_list), 1)
		for j in range(0, len(hidden_list)):
			feat_norm = torch.norm(fc_1) + eps
			state = hidden_list[j][hidden_list[j].shape[1]-2].squeeze(1).detach()
			hid_norm = torch.norm(state) + eps	
			fc_1 = fc_1.view(1, self.hidden_dim).to(device)
			state = state.view(self.hidden_dim, 1).to(device)
			reward[j] = torch.mm(fc_1, state)/(feat_norm*hid_norm)
		Loss = torch.zeros(len(hidden_list), 1)
		#Calculating loss
		for j in range(0, len(hidden_list)):
			t_1 = torch.mm(hidden_list[j][hidden_list[j].shape[1]-2].squeeze(1).detach().to(device), fc_1.view(fc_1.shape[1],1))
			t = torch.mm(hidden_list[j][hidden_list[j].shape[1]-1].squeeze(1).detach().to(device), fc_1.view(fc_1.shape[1],1))
			Loss[j] = max(0, t_1 - t)*0.9
		Loss = self.alpha*(Loss + reward)
		return Loss, reward


class ValueNet(nn.Module):
	def __init__(self, embed_size, vocab_size, hidden_size, vocab, max_seq):
		super(ValueNet, self).__init__()
		self.embed_size = embed_size
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.vocab = vocab
		self.CNNv = EncoderCNN(embed_size)
		self.RNNv = DecoderRNN(vocab_size, embed_size, hidden_size, vocab, max_seq)
		self.fc1 = nn.utils.weight_norm(nn.Linear(embed_size*2 , embed_size))
		self.fc2 = nn.utils.weight_norm(nn.Linear(embed_size, embed_size))
		self.fc3 = nn.utils.weight_norm(nn.Linear(embed_size, 1))
		self.relu = nn.LeakyReLU(0.2, inplace = True)
		self.norm1 = nn.LayerNorm(embed_size)
		self.norm2 = nn.LayerNorm(embed_size)

	def features_extract(self,images):
		features = self.CNNv(images)
		features = features.unsqueeze(1)
		return features
	#Gives value function for each generated word for the image
	def forward(self, images, generated_embed):
		features = self.features_extract(images)
		generated_embed = generated_embed.to(device)
		bs, max_seq,_, _, _ = generated_embed.shape
		value = torch.zeros(generated_embed.shape[1], 1)
		in_features = features.to(device)
		for index in range(0, generated_embed.shape[1]):
			captions_in = generated_embed[:, index, :, :, :]
			# captions_in = captions_in.view(bs, 1, -1).to(device)
			captions_in = captions_in.squeeze(0)
			input = torch.cat((in_features, captions_in), 2)
			fc_1 = self.relu(self.norm1(self.fc1(input)))
			fc_1 = self.relu(self.norm2(self.fc2(fc_1)))
			fc_1 = torch.tanh(self.fc3(fc_1))
			value[index] = fc_1
			in_features = generated_embed[:, index, :, :, :].squeeze(0)
		return value

	#Loss for this network
	def loss(self, image, reward, captions):
		value = self.forward(image, captions)
		value = value.to(device)
		reward = reward.to(device)
		Loss = F.smooth_l1_loss(value, reward)
		return Loss	

	#embedding features of Rnnv in hidden state of original captions
	def captions_hidden_state(self, caption):
		in_captions = self.RNNv.embed(caption)
		hiddens, hiddens_list = self.RNNv.forward_captions(in_captions)
		return hiddens, hiddens_list
		








