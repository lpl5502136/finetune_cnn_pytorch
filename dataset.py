#coding=utf-8
from __future__ import print_function, division
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
from config import *


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# use PIL Image to read image
def read_image(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class CustomData(Dataset):
    def __init__(self, csv_file, data_transform=None):
        self.img_path_list = []
        self.label_list = []
        self.data_transform = data_transform
        lines = open(csv_file).readlines()
        for line in lines:
            ws = line.strip("\n").split(",")
            labels = ws[1]
            path = ws[0]
            self.img_path_list.append(path)
            self.label_list.append(labels)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        img_name = self.img_path_list[item]
        label = int(self.label_list[item])
        img = read_image(img_name)
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, label


if __name__ == '__main__':
    dir = "./Desktop/hymenoptera_data"
    # customData()
