import torch
from torch import nn
from torch.autograd import Variable
from settings import config

class PairwiseRankingLoss(nn.Module):
	def __init__(self):
		super(PairwiseRankingLoss, self).__init__()
		self.margin = config["margin_pairwise_ranking_loss"]

	def forward(self, sentence, image):				
		margin = self.margin		
		scores = torch.mm(image, sentence.transpose(1, 0))
		diagonal = scores.diag()

		sentence_cost = None		
		image_cost = None

		if torch.cuda.is_available() and config["cuda"] == True:
			sentence_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])).cuda(), (self.margin-diagonal).expand_as(scores)+scores)		
			image_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])).cuda(), (self.margin-diagonal).expand_as(scores).transpose(1, 0)+scores)
		else:
			sentence_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])), (margin-diagonal).expand_as(scores)+scores)
			image_cost = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1])), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

		for i in range(scores.size()[0]):
			sentence_cost[i, i] = 0
			image_cost[i, i] = 0

		return sentence_cost.sum() + image_cost.sum()