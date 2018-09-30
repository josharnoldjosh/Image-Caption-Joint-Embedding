from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

import torch

if __name__ == "__main__":

	# Load data
	data = Data(batch_size=config["batch_size"])

	# Load model
	model = Model(config)

	# Model loss function
	loss = Loss()
	
	# Optimizer 
	optimizer = Optimizer(model)

	for epoch in range(config["num_epochs"]):

		# Each epoch
		print("Starting epoch", epoch+1)		

		for caption_sequence, image_feature in data:
			
			# Pass data through model
			caption_sequence, image_feature = model(caption_sequence, image_feature)

			# Compute loss
			cost = loss(caption_sequence, image_feature)

			# Optimize loss and perform back-propagation
			optimizer.backprop(cost)

			# Computing results 
			

	print("Script done.")