from data import Data
from model import Model

def main():

	data = Data(test=True)
	model = Model(data)
	model.evaluate(data)
	print("\nScript done :)")
	return

if __name__ == '__main__':
	main()