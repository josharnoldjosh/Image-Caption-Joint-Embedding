import torch

class Optimizer:
	def __init__(self, model, learning_rate=0.0002):
		self.learning_rate = learning_rate
		self.params = filter(lambda p: p.requires_grad, model.parameters())
		self.optimizer = torch.optim.Adam(self.params, learning_rate)
		
	def backprop(self, cost, grad_clip=2.0):
		print("	* Cost:", cost.data.cpu().numpy())
		cost.backward()
		torch.nn.utils.clip_grad_norm_(self.params, grad_clip)
		self.optimizer.step()

