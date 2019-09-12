from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from gcommand_loader import GCommandLoader
import numpy as np
from Model import FirstNet, VGG
from Trainer import train, test_validation
import os

# Training settings
train_path ='data/train'

test_path ='data/test'

valid_path ='data/valid'

batch_size = 100
test_batch_size=100
arc = 'FirstNet'
epochs = 100
lr = 0.001
momentum=0.9
optimizer='adam'
cuda=True
seed=1234
log_interval=10
patience=5

# feature extraction options
window_size=.02
window_stride=.01
window_type='hamming'
normalize=True

cuda = torch.cuda.is_available()

# loading data
train_dataset = GCommandLoader(train_path, window_size=window_size, window_stride=window_stride,
                               window_type=window_type, normalize=normalize)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=20, pin_memory=cuda, sampler=None)

valid_dataset = GCommandLoader(valid_path, window_size=window_size, window_stride=window_stride,
                               window_type=window_type, normalize=normalize)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=None,
    num_workers=20, pin_memory=cuda, sampler=None)

test_dataset = GCommandLoader(test_path, window_size=window_size, window_stride=window_stride,
                              window_type=window_type, normalize=normalize)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=20, pin_memory=cuda, sampler=None)

# build model
if arc == 'FirstNet':
    model = FirstNet()
elif arc.startswith('VGG'):
    model = VGG(arc)
else:
    model = FirstNet()

if torch.cuda.is_available():
    model.cuda()

# define optimizer
if optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1

# args.epochs + 1
# trainint with early stopping
while (epoch <= 5) and (iteration < patience):
    train(train_loader, model, optimizer, epoch, cuda, log_interval)
    valid_loss = test_validation(valid_loader, model, cuda)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        iteration = 0
        best_valid_loss = valid_loss
    epoch += 1

# write test y
model.eval()
predictions = []
for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    outputs = model(data)
    predicts = torch.max(outputs.data, 1)[1]
    predictions.extend(predicts.tolist())

with open('test_y', 'w') as test_y:
    for prediction, filename in zip(predictions, test_loader.dataset.spects):
      test_y.write("{}, {}\n".format(os.path.basename(filename[0]), prediction))

