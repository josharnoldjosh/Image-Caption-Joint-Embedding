from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

if __name__ == "__main__":

	# Load data
	data = Data()
	data.load_dictionaries()
	
	# Load model
	model = Model(data)
	model.input_name = "best"
	model.output_name = "final"
	model.load()

	# Model loss function
	loss = Loss()

	# Optimizer 
	optimizer = Optimizer(model)

	# Begin epochs
	for epoch in range(config["num_epochs"]):
		print("[EPOCH]", epoch+1)

		# Process batches
		for caption, image_feature in data:
			pass			

			# Pass data through model
			caption, image_feature = model(caption, image_feature)

			# Compute loss
			cost = loss(caption, image_feature)			

			# Zero gradient, Optimize loss, and perform back-propagation
			optimizer.backprop(cost)

		# Evaluate final model results					
		model.evaluate(data, save_if_better=True)

	# Final evaluation	
	print("\nFinal evaluation:")
	model.evaluate(data, save_if_better=True)		
	
	print("\n[SCRIPT] complete")