import torch
from settings import config

class Optimizer:
	def __init__(self, model, learning_rate=0.0002):
		self.learning_rate = config["learning_rate"]
		self.params = filter(lambda p: p.requires_grad, model.parameters())
		self.optimizer = torch.optim.Adam(self.params, learning_rate)
		
	def backprop(self, cost, grad_clip=2.0):
		print("	* Cost:", cost.data.cpu().numpy())		
		self.optimizer.zero_grad() # Reset gradient
		cost.backward() # Back propagate
		torch.nn.utils.clip_grad_norm(self.params, grad_clip)
		self.optimizer.step()

