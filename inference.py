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
import sys


if __name__ == '__main__':
    """
    python inference.py /root/code/finetune_cnn_pytorch/output/resnet_epoch9.pth
    """
    model_weight = sys.argv[1]
    data_transform = data_transforms['val']

    # get model and replace the original fc layer with your fc layer
    model_ft = get_resnet18(num_class, use_pretrained=True)
    model_ft.load_state_dict(torch.load(model_weight))
    # model_ft = torch.load(model_weight)
    model_ft.eval()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    lines = open("./scripts/val.csv").readlines()
    right = 0
    count = 0
    for line in lines:
        path, label = line.strip("\n").split(",")
        img = read_image(path)
        if data_transform is not None:
            img = data_transform(img)
        inputs = Variable(img.unsqueeze(0).cuda())
        # print(inputs.data)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs.data, 1)
        print(label, preds[0])
        if int(label) == int(preds[0]):
            right += 1
        count += 1

    print("precison:{}/{}--{}".format(right, count, right/(count+0.0001)))

