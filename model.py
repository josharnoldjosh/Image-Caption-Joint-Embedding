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

	# COMBINE X AND Y TO TEST ALL OF DEV DATA ISNTEAD OF JUST ONE BATCH
	def encode(self, data):
		data.use_dev = True
		old_batch = data.batch_number
		data.batch_number = 0
		captions = None
		images = None
		for caption, image_feature in data:				
			x, y = self.forward(caption, image_feature)		
			captions = x
			images = y	
			# if captions is None:
			# 	captions = x				
			# else:
			# 	captions = torch.cat((captions, x), 0) 

			# if images is None:
			# 	images = y				
			# else:
			# 	images = torch.cat((images, y), 0)

		print("	* encoded data")
		data.use_dev = False		
		data.batch_number = old_batch
					
		return captions, images

	def evaluate(self, data):
		print("	* Validating...")
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