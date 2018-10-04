from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

if __name__ == "__main__":

	# Load data
	data = Data()

	# Load model
	model = Model(data)

	# Model loss function
	loss = Loss()
	
	# Optimizer 
	optimizer = Optimizer(model)

	for epoch in range(config["num_epochs"]):		
		print("\nStarting epoch", epoch+1)		
		validation_freq = config["validation_freq"]
		validation_count = 0
				
		for caption, image_feature in data:		

			validation_count += 1			

			# Pass data through model
			caption, image_feature = model(caption, image_feature)

			# Compute loss
			cost = loss(caption, image_feature)			

			# Zero gradient, Optimize loss, and perform back-propagation
			optimizer.backprop(cost)

			# Evaluate results & save best model
			if validation_count % validation_freq == 0:	
				print("	* Validating...")
				model.evaluate(data)			

	print("Script done.")