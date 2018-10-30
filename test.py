from data import Data
from model import Model

def main():
	data = Data(test=True)	
	model = Model(data)
	model.input_name = "transfer"
	model.load()
	model.evaluate(data)	
	return

if __name__ == '__main__':
	main()
	print("\nScript done :)")