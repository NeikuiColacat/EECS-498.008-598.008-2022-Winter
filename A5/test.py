import torch.nn as nn
import torch 
import math

a = torch.tensor([[1,2,3],[4,5,6]] , dtype=torch.float16)
print(a.shape)
a = torch.unsqueeze(a,dim=1)
print(a.shape)

