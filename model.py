import torch
import torch.nn.functional as F
from settings import config
import numpy
import evaluate
from collections import defaultdict
import time

class Model(torch.nn.Module):
	def __init__(self, data):
		super(Model, self).__init__()

		# Performance score
		self.score = 0

		# number of words in dictionary
		num_words = len(data.word_to_index)

		# Sentence
		self.embedding = torch.nn.Embedding(num_words, config['word_dimension'])
		self.lstm = torch.nn.LSTM(config['word_dimension'], config['model_dimension'], 1)

		# Image - Assume image feature is already extracted from pre-trained CNN
		self.linear = torch.nn.Linear(config['image_dimension'], config['model_dimension'])

		# Initialize weights for linear layer
		torch.nn.init.xavier_uniform_(self.linear.weight)		
		self.linear.bias.data.fill_(0)		

		if torch.cuda.is_available() and config["cuda"] == True:
			self.embedding.cuda()
			self.lstm.cuda()
			self.linear.cuda()

	def forward(self, sentence, image):		
		return self.forward_caption(sentence), self.forward_image(image)

	def forward_image(self, image):
		# Pass image through model
		image_embedding = self.linear(image)

		# Normalize
		norm_image_embedding = F.normalize(image_embedding, p=2, dim=1)

		return norm_image_embedding

	def forward_caption(self, sentence):

		# Pass caption through model
		sentence_embedding = self.embedding(sentence)

		_, (sentence_embedding, _) = self.lstm(sentence_embedding)

		x_sentence_embedding = sentence_embedding.squeeze(0)

		# Normalize vectors
		norm_sentence_embedding = F.normalize(x_sentence_embedding, p=2, dim=1)		

		return norm_sentence_embedding

	def evaluate(self, data, verbose=False):
		print("	* Validating...")			
		data.set_dev_mode(True)
		score = 0
		for caption, image_feature in data:				
			x, y = self.forward(caption, image_feature)
			score_1 = evaluate.image_to_text(x, y, verbose=verbose)
			score_2 = evaluate.text_to_image(x, y, verbose=verbose)	
			score += (score_1 + score_2)			
		
		data.set_dev_mode(False)
		print("	* Scored %.2f" % score)		
		self.save(score)
		return

	def save(self, score):
		if score > self.score:
			self.score = score
			print('	* Saving model...')			
			torch.save(self.state_dict(), 'best.pkl')
			print('	* Done!')
		return