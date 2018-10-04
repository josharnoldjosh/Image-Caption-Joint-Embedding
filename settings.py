config = {
	"num_epochs":100,
	"batch_size":128,
	"num_words":10000, # Ignore sentences longer than this
	"word_dimension":1000, # The dimensionality of word embeddings
	"image_dimension":4096, # input image dimension
	"model_dimension":1000, # The dimension of the embedding space,
	"learning_rate":0.001,
	"margin_pairwise_ranking_loss":0.2 # Should be between zero and 1
}