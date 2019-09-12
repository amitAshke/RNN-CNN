from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import os

def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data
        if False:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.data))
    return global_epoch_loss / len(loader.dataset)

def test_validation(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nValidation test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss

#
# def test(loader, model, cuda, verbose=True):
#     model.eval()
#     correct = 0
#     predictions = []
#     for data, target in loader:
#         if torch.cuda.is_available():
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         output = model(data)
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#         predictions.append(correct)
#     return predictions
#
# # def write_test_y(loader, all_test_predictions):
# #     file = open('test_y', 'w')
# #     for pred, filename in zip(all_test_predictions, loader.dataset.spects):
# #         file.write("{}, {}\n".format(os.path.basename(filename[0]), pred))
# #     file.close()
#
# def write_test_y(model, test_data):
#     model.eval()
#     predictions = []
#     for data, target in test_data:
#         outputs = model(data)
#         predicts = torch.max(outputs.data, 1)[1]
#         predictions.extend(predicts.tolist())
#
#     with open('test_y', 'w') as test_y:
#         for prediction, filename in zip(predictions, test_data.dataset.spects):
#             test_y.write("{}, {}\n".format(os.path.basename(filename[0]), prediction))