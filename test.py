import torch
import torchvision.transforms as transforms

from models import UNet
from datasets import clinic

import math
import numpy as np
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# model
model = UNet.UNet()
# model.load_state_dict(torch.load('./checkpoints/?.pth')['state_dict'])
model.load_state_dict(torch.load('./trained_models/UNet_clinic'))
model.to(device)
model.eval()


# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("model param num: {}".format(count_parameters(model)))


# Visualization Functions
def tensor_to_numpy(tensor):
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img


#########################################################
# Example Data Visualization
# Clinic
input_path = './Clinic_dataset/input_image'
label_path = './Clinic_dataset/binary_image'
input_files = sorted(glob(os.path.join(input_path, '*.tif')))
label_files = sorted(glob(os.path.join(label_path, '*.tif')))

# Transforms
input_transform = transforms.Compose([
    transforms.ToTensor()
])

label_transform = transforms.Compose([
    transforms.ToTensor()
])

# Numpy
index = 0
input_numpy = tiff.imread(input_files[index])
label_numpy = tiff.imread(label_files[index])
label_numpy = np.expand_dims(label_numpy, axis=2)

# Tensor with transforms
input_tensor = input_transform(input_numpy).to(device)
label_tensor = label_transform(label_numpy)
output_tensor = model(input_tensor.unsqueeze(0))

# Tensor to Numpy
input_image = tensor_to_numpy(input_tensor.cpu())
label_image = tensor_to_numpy(label_tensor)
output_image = tensor_to_numpy(output_tensor.squeeze(0).cpu())

# Show Image with plt
fig = plt.figure()
rows = 1
cols = 3

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(input_image)
ax1.set_title('input')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(output_image)
ax2.set_title('output')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(label_image)
ax3.set_title('label')
ax3.axis("off")

plt.tight_layout()
plt.show()
