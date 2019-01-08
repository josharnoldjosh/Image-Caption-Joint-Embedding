config = {

	# Input parameters - BE CAREFUL WITH THESE! Changing to 'bad' value can break script
	"dataset":"visual_fashion_dialog", # visual_fashion_dialog, deepfashion
	"image_dimension":512, # input image dimension : 4096 for VGG, 512 resnet

	# Training parameters	
	"num_epochs":50,
	"batch_size":64, # this can't be too small
	"learning_rate":0.001, # learning rate
	"display_freq":4, # how often to display loss : 1 = every batch, 2 = every second batch etc...

	# Specify dimensions - you can choose whatever you want for this
	"word_dimension":128, # The dimensionality of word embeddings / max sequence to encode
	"model_dimension":1000, # The dimension of the embedding space,	
	"lstm_dialog_emb":1000, # the hidden vec output size of the LSTM

	# Loss function
	"margin_pairwise_ranking_loss":0.2, # Should be between zero and one

	# CUDA
	"cuda":True # enable cuda
}