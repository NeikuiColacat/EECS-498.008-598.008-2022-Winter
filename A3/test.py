import torch

t = torch.tensor([1,2,3])
tmp = {'a' : t}

re = tmp['a']

re[0] = 10

print(tmp['a'])

