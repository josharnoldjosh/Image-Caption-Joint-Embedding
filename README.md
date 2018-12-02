# Image Caption Joint Embedding

A multimodal embedding of images and captions, built with PyTorch, written with Python3.

## Introduction

Inspired by the paper: https://github.com/linxd5/VSE_Pytorch

## Getting started

Place your data in the `data/<dataset_name>/` folder. To find out more about preprocessing your data, please check out this repository: https://github.com/josharnoldjosh/Deep-Fashion-Joint-Embedding-Preprocessing The repository formats a popular dataset to train as a joint embedding. You can follow the code, it is fairly easy to understand.

In addition, make sure you have an empty folder called `dict`.
    
## Train the model

Run `python3 train.py` to train your model. Run `python3 test.py` to test your model.

Alot of work has been done behind the scenes. As long as your input data is formatted properly, everything should be good. The only three files you will need to worry about is `train.py`, `test.py`, and `settings.py`. 

The script has been setup to use k-fold cross validation. Make sure you configure `settings.py`. Everything else should be straight forward.

Training your model is this easy:

```python
from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer

# Load data
data = Data()

# track score to save best model
score = 0

# Use K fold cross validation for model selection
for train, test, fold in data.k_folds(5):		

	# Prepare data to use the current fold
	data.process(train, test, fold)

	# Load model
	model = Model(data)

	# Model loss function
	loss = Loss()

	# Optimizer 
	optimizer = Optimizer(model)

	# Begin epochs
	for epoch in range(config["num_epochs"]):		

		# Process batches
		for caption, image_feature in data:
			pass			

			# Pass data through model
			caption, image_feature = model(caption, image_feature)

			# Compute loss
			cost = loss(caption, image_feature)			

			# Zero gradient, Optimize loss, and perform back-propagation
			optimizer.backprop(cost)

		# Evaluate model results					
		model.evaluate(data)

	# Final evaluation - save if results are better			
	model_score = model.evaluate(data)
	if model_score > score:
		score = model_score
		model.save()
		data.save_dictionaries()		
```