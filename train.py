#coding=utf-8
from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models, transforms
import time
import os
from torch.utils.data import Dataset
from PIL import Image

from dataset import *
from config import *
from model import *


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            count_batch = 0
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                count_batch += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                # running_loss += loss.data[0]
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

                # print result every 10 batch
                if count_batch % 10 == 0:
                    batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*count_batch)
                    print('{} Epoch [{}] Batch [{}] Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'. \
                          format(phase, epoch, count_batch, batch_loss, batch_acc, time.time()-begin_time))
                    begin_time = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model_ft.state_dict(), "output/resnet_epoch{}_{}.pth".format(epoch, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return


if __name__ == '__main__':
    # create dataset and dataloader
    image_datasets = {x: CustomData("./scripts/{}.csv".format(x), data_transform=data_transforms[x]) for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # get model and replace the original fc layer with your fc layer
    model_ft = get_resnet18(num_class, use_pretrained=True)
    # model_ft.load_state_dict(torch.load("/root/code/finetune_cnn_pytorch/output/best_resnet.pkl"))

    # if use gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # multi-GPU
    # model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    train_model(model=model_ft,
               criterion=criterion,
               optimizer=optimizer_ft,
               scheduler=exp_lr_scheduler,
               num_epochs=num_epochs,
               use_gpu=use_gpu)

    # save best model
    # torch.save(model_ft, "output/best_resnet.pkl")
    torch.save(model_ft.state_dict(), "output/best_model.pth")
