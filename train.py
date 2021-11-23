import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import random
import argparse
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datasets import clinic
from models import UNet

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Random Seed
seed = random.randint(1, 10000)
torch.manual_seed(seed)

# Training parameters
parser = argparse.ArgumentParser(description="PyTorch SR")
parser.add_argument("--model_name", type=str, default='UNet_clinic', help="model_name")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--step", type=int, default=200, help="Learning Rate Scheduler Step")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay")
opt = parser.parse_args()

# Dataset
clinic_dataset = clinic.ClinicDataset()
dataset = clinic_dataset
print(len(dataset))
train_validation_dataset, test_dataset = train_test_split(dataset, train_size=500, shuffle=False)
train_dataset, validation_dataset = train_test_split(train_validation_dataset, train_size=400, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=opt.batchSize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)

# Model
model = UNet.UNet()
model.to(device)

# Optimizer
def dice_loss(outputs, labels):
    outputs = outputs.contiguous()
    labels = labels.contiguous()

    intersection = (outputs * labels).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + 1) / (outputs.sum(dim=2).sum(dim=2) + labels.sum(dim=2).sum(dim=2) + 1)))

    return loss.mean()


def calc_loss(outputs, labels):
    bce = F.binary_cross_entropy_with_logits(outputs, labels, reduction='mean')
    dice = dice_loss(F.sigmoid(outputs), labels)

    loss = bce + dice
    return loss

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.99), eps=1e-08)


# Learning rate scheduling
def adjust_learning_rate(epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

# Checkpoint directory
checkpoint_path = './checkpoints/'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_loss_list = []
validation_loss_list = []

# Training
def train(model, optimizer, epoch):
    # Learning rate scheduler
    lr = adjust_learning_rate(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Training
    model.train()

    for batch, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        loss = calc_loss(model(inputs), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * opt.batchSize, len(train_dataloader.dataset),
                       100. * batch / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())

    # Validation
    val_loss = 0
    model.eval()

    for batch, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        loss = calc_loss(model(inputs), labels)
        val_loss += loss.item()

        if batch % 10 == 0:
            validation_loss_list.append(loss.item())

    print('Validation Epoch: {} Loss: {:.6f}'.format(
        epoch, val_loss / len(validation_dataloader)))

    # checkpoint
    if epoch % 5 == 0:
        checkpoint = {'model': model,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, os.path.join(checkpoint_path, opt.model_name + '_' + str(epoch) + '.pth'))

# Training
start = time.time()

for epoch in range(1, opt.nEpochs + 1):
    train(model, optimizer, epoch)

# Save Model
torch.save(model.state_dict(), os.path.join('./trained_models/', opt.model_name))

# Save Time
print("time :", time.time() - start)
history = open('runtime history.txt', 'a')
history.write('model: {}, epoch: {}, batch: {}, runtime: {} \n'.format(opt.model_name, opt.batchSize, opt.nEpochs, time.time() - start))

# Visualize loss
df_train = pd.DataFrame(train_loss_list)
df_validation = pd.DataFrame(validation_loss_list)

fig, axs = plt.subplots(2)
axs[0].plot(df_train)
axs[0].set_title('train loss')

axs[1].plot(df_validation)
axs[1].set_title('validation loss')

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'Graphs', opt.model_name + '_loss graph.png'))
plt.show()
