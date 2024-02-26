import torch
import numpy as np
from torch.autograd.functional import hessian
import torch.nn.functional as F

class Hessian:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def set_y(self, y):
        self.y = y

    def get_hessian(self, x):
        logit = self.model(x.reshape(1, 3, 32, 32))
        loss = F.cross_entropy(logit, self.y)
        return loss
    
    def get_eigenvalue(self, matrix):
        L = torch.linalg.eigvals(matrix)
        return max(torch.abs(L))