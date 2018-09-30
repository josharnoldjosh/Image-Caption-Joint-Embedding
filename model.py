import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()

		# Sentence
		self.embedding = torch.nn.Embedding(config['num_words'], config['word_dimension'])
		self.lstm = torch.nn.LSTM(config['word_dimension'], config['model_dimension'], 1)

		# Image - Assume image feature is already extracted from pre-trained CNN
		self.linear = torch.nn.Linear(config['image_dimension'], config['model_dimension'])

	def forward(self, sentence, image):		
		# Pass image through network
		image_embedding = self.linear(image)		

		# Pass the sentence through the models network
		sentence_embedding = self.embedding(sentence)
		_, (sentence_embedding, _) = self.lstm(sentence_embedding)
		sentence_embedding = sentence_embedding.squeeze(0)

		# Normalize vectors
		norm_sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
		norm_image_embedding = F.normalize(image_embedding, p=2, dim=1)

		return norm_sentence_embedding, norm_image_embedding

