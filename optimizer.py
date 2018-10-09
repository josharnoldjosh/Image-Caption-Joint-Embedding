import torch
from settings import config

class Optimizer:
	def __init__(self, model, learning_rate=0.0002):
		self.learning_rate = config["learning_rate"]
		self.params = filter(lambda p: p.requires_grad, model.parameters())
		self.optimizer = torch.optim.Adam(self.params, learning_rate)
		self.display_freq = config["display_freq"]
		self.display_count = 0
		
	def backprop(self, cost, grad_clip=2.0):		
		if self.display_count % self.display_freq == 0 or self.display_count == 0:
			print("	* Cost:", cost.data.cpu().numpy())		
		self.display_count += 1
		self.optimizer.zero_grad() # Reset gradient
		cost.backward() # Back propagate		
		torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
		self.optimizer.step()