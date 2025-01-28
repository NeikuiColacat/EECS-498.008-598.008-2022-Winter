import torch.nn as nn
import torch 
import math

a = torch.tensor([[1,2,3],[4,5,6]] , dtype=torch.float16)
a = a.reshape(2 , 1 , 3)

a = torch.nn.functional.softmax(a , dim = 2)
print(a)

