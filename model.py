import torch
import torch.nn.functional as F
from settings import config
import numpy
import evaluate
from collections import defaultdict
import time
import deepdish as dd

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		# Performance score
		self.score = 0

		# Sentence
		self.embedding = torch.nn.Embedding(config['num_words'], config['word_dimension'])
		self.lstm = torch.nn.LSTM(config['word_dimension'], config['model_dimension'], 1)

		# Image - Assume image feature is already extracted from pre-trained CNN
		self.linear = torch.nn.Linear(config['image_dimension'], config['model_dimension'])

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
		sentence_embedding = sentence_embedding.squeeze(0)

		# Normalize vectors
		norm_sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)		

		return norm_sentence_embedding

	def encode(self, data):			
		captions, images = data.preprocess_data(data.dev[0], data.dev[1])
		return self.forward(captions, images)

	def evaluate(self, data):
		r_time = time.time()
		encoded_captions, encoded_images = self.encode(data)
		score_1 = evaluate.image_to_text(encoded_captions, encoded_images)
		score_2 = evaluate.text_to_image(encoded_captions, encoded_images)		
		print("		* Recall@K in %.1fs" % float(time.time()-r_time))
		self.save(score_1+score_2)		

	def save(self, score):
		if score > self.score:
			self.score = score
			print('		* Saving model...')			
			torch.save(self.state_dict(), 'best.pkl')
			print('		* Done!')
		return