from data import Data
from model import Model

def main():

	data = Data() # REMEMBER to update the dataset name in settings.py to the test set if you have one
	data.load_dictionaries() # very important - load dictionaries 

	model = Model(data)
	model.input_name = "best" # specify save model name
	model.load() # load the saved model weights

	model.evaluate(data) # evaluate the data

	return

if __name__ == '__main__':
	main()
	print("\n[DONE] script finished")