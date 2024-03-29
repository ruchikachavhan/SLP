import nltk
import pickle
import argparse
from collections import Counter
import json

class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self):
		#need two dictionaries for easy access of words and indices
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		#if that particular word is not present then add and update id
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	#Constructor
	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)

	def print_vocab(self):
		print(self.idx2word)

def build_vocab(json_file, threshold):
	file = open(json_file, "r")
	data = json.loads(file.read())
	counter = Counter()
	#There are two ids in the keys, images and dataset. Dataset is just a string
	for i, id in enumerate(data.keys()):
		for j in range(0, len(data['images'])):
			name = data['images'][j]['filename'].split('_')[0]
			# or name == 'commercial' or name == 'stadium'
			if( name == 'airport' or name == 'parking' or name == 'baseballfield' ):
				sentence = data['images'][j]['sentences'][3]['raw'] #For now training with only first sentence id 
				tokens = nltk.tokenize.word_tokenize(sentence)
				counter.update(tokens)

	print("[{}/{}] Tokenized the captions.".format(i+1, len(data.keys())))
	
	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]
	words = [word.lower() for i, word in enumerate(words) if word.isalpha()]

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)
	vocab.print_vocab()
	return vocab

def main(args):
    vocab = build_vocab(json_file=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/home/ruchika/RSICD/annotations_RSICD/dataset_rsicd.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/home/ruchika/RSICD/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)