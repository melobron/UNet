import torch
import torch.nn as nn


input = torch.Tensor(2, 10, 99, 99)
model1 = nn.Conv2d(10, 20, 3)
pool = nn.MaxPool2d(2)
model2 = nn.ConvTranspose2d(20, 10, 2, 2)
output1 = pool(model1(input))
output2 = model2(output1)
print(output2.shape)
