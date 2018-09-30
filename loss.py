import torch
from torch import nn
from torch.autograd import Variable

class PairwiseRankingLoss(nn.Module):
	def __init__(self, margin=1.0):
		super(PairwiseRankingLoss, self).__init__()
		self.margin = margin

	def forward(self, sentence, image):
		sentence = sentence.transpose(1, 0)
		scores = torch.mm(image, sentence)
		diagonal = scores.diag()

		sentence_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])), (self.margin-diagonal).expand_as(scores)+scores)		
		image_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])), (self.margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

		if torch.cuda.is_available():
			sentence_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])).cuda(), (self.margin-diagonal).expand_as(scores)+scores)		
			image_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])).cuda(), (self.margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

		for i in range(scores.size()[0]):
			sentence_cost[i, i] = 0
			image_cost[i, i] = 0

		return sentence_cost.sum() + image_cost.sum()