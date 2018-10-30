from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

if __name__ == "__main__":

	# Load data (plus load dictionaries)
	data = Data(load_dict=True)

	# Load model
	model = Model(data)
	model.load() # load weights
	model.output_name = "transfer" #specify output file so we don't overwrite best.pkl

	# Model loss function
	loss = Loss()
	
	# Optimizer 
	optimizer = Optimizer(model)
	
	for epoch in range(config["num_epochs"]):		
		print("\n[EPOCH]", epoch+1)				
				
		for caption, image_feature in data:					

			# Pass data through model
			caption, image_feature = model(caption, image_feature)

			# Compute loss
			cost = loss(caption, image_feature)			

			# Zero gradient, Optimize loss, and perform back-propagation
			optimizer.backprop(cost)

		# Evaluate results & save best model					
		model.evaluate(data)

	# Final evaluation			
	print("\n[EVAL] result:")
	model.evaluate(data)

	print("Script done.")