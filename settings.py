config = {
	"num_epochs":15,
	"batch_size":128,
	"word_dimension":1000, # The dimensionality of word embeddings
	"image_dimension":4096, # input image dimension
	"model_dimension":1000, # The dimension of the embedding space,
	"learning_rate":0.001,
	"validation_freq":10,
	"display_freq":1,
	"margin_pairwise_ranking_loss":0.2 # Should be between zero and 1
}