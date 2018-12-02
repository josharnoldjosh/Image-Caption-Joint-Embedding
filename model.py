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

		# Filename
		self.input_name = "best"
		self.output_name = "best"

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

	def average_i2t_and_t2i(self, i2t, t2i):
		i_r1, i_r5, i_r10, i_medr, t_r1, t_r5, t_r10, t_medr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

		for x in i2t:
			i_r1 += x[0]
			i_r5 += x[1]
			i_r10 += x[2]
			i_medr += x[3]

		for x in t2i:
			t_r1 += x[0]
			t_r5 += x[1]
			t_r10 += x[2]
			t_medr += x[3]

		i_r1 = i_r1/len(i2t)
		i_r5 = i_r5/len(i2t)
		i_r10 = i_r10/len(i2t)
		i_medr = i_medr/len(i2t)

		t_r1 = t_r1/len(i2t)
		t_r5 = t_r5/len(i2t)
		t_r10 = t_r10/len(i2t)
		t_medr = t_medr/len(i2t)

		print("	* Image to text scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (i_r1, i_r5, i_r10, i_medr))		
		print("	* Text to image scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (t_r1, t_r5, t_r10, t_medr))
			
		return

	def evaluate(self, data, verbose=False):
		"""
		If using k-fold cross validation in the data module,
		the data class will handle updaing the self.train and self.test
		datasets. Thus the data.test_set(True) becomes very important.
		However, a raw intialization of the dataclass with result in
		the loaded data being assigned to both test and train, so we can
		evaluate the results. 
		"""
		print("	* Validating", end="", flush=True)				
		data.test_set(True) # very important | swaps to iterating over the test set for validation
		score = 0
		i2t, t2i = [], []
		for caption, image_feature in data:				
			x, y = self.forward(caption, image_feature)
			score_1, i2t_result = evaluate.image_to_text(x, y, verbose=verbose)
			score_2, t2i_result = evaluate.text_to_image(x, y, verbose=verbose)	
			score += (score_1 + score_2)		
			i2t.append(i2t_result)	
			t2i.append(t2i_result)
			print(".", end="", flush=True)		
		
		print("[DONE]", end="", flush=True)
		print("")
		data.test_set(False) # also very important | swaps BACK to using the TRAIN set
		self.average_i2t_and_t2i(i2t, t2i)
		return score	

	def save(self):
		print('	* Saving model...')			
		torch.save(self.state_dict(), self.output_name+'.pkl')
		print('	* Done!')
		return

	def load(self):		
		self.load_state_dict(torch.load(self.input_name+".pkl"))
		print("[LOADED]", self.input_name+".pkl", "\n")
		return