import cv2
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
import random


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = 'eff_img' + os.sep, 'eff_img_label' + os.sep  # crop_img/, crop_label/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
 
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


class TextDataset(Dataset):
    def __init__(self, imgs_path, img_size = 64):
        self.imgs_path = imgs_path
        with open(self.imgs_path, 'r') as t:
            self.imgs_file = t.read().strip().splitlines()
            t.close()
        self.labels_file = img2label_paths(self.imgs_file)
        self.indices = range(len(self.imgs_file))

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_file[idx])
        img = cv2.resize(img, (img_size, img_size))
        img = rotate(img, random.randint(-90, 90))
        if type(img) == type(None):
            print(self.imgs_file[idx])
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        with open(self.labels_file[idx], 'r', encoding='utf-8') as f:
            temp = [int(f.read().strip().splitlines()[0])]
            label = torch.LongTensor(temp)
        return img, label

    def __len__(self):
        return len(self.imgs_file)


class TextModel(nn.Module):
    def __init__(self, num_classes):
        super(TextModel, self).__init__()
        self.extract = EfficientNet.from_pretrained('efficientnet-b7')
        self.extract._fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.extract(x)
        return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    valid = False
    word_to_num = json.load(open('new_word_to_num.json', 'r', encoding='utf-8'))
    num_classes = len(word_to_num)
    # hyperparameters
    img_size = 64
    batch_size = 256
    lr = 1e-3
    epoch = 50
    # create dataset
    if valid:
        train_set = TextDataset(os.path.normcase('new_eff_train.txt'), img_size)
        val_set = TextDataset(os.path.normcase('new_eff_val.txt'), img_size)
    else:
        train_set = TextDataset(os.path.normcase('new_eff_total.txt'), img_size)
    # create dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    if valid:
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # create model
    model = TextModel(num_classes)
    print(model)
    '''
    # freeze feature extract module
    for name, param in model.named_parameters():
        if name.find('extract') != -1:
            param.requires_grad = False
        # print(name, param.requires_grad)
    '''
    # optimizer and loss function
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.33)
    loss_fn = nn.CrossEntropyLoss()
    # start training
    model.to(device)
    model.train()
    training_loss = []
    val_loss = []
    for i in range(epoch):
        with tqdm(train_loader, unit='batch') as tepoch:
            num_target = 0
            correct = 0
            for img, label in tepoch:
                tepoch.set_description('Epoch %d' % i)
                img = Variable(img.to(device))
                label = Variable(label.to(device)).squeeze()
                optimizer.zero_grad()
                pred = model(img)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())
                num_target += len(label)
                correct += (pred.argmax(dim=1) == label).sum().item()
                tepoch.set_postfix(loss=loss.item(), train_accuracy=correct / num_target)
        if valid:
          with tqdm(val_loader, unit='batch') as tepoch:
              num_target = 0
              correct = 0
              for img, label in tepoch:
                  tepoch.set_description('Validation')
                  img = Variable(img.to(device))
                  label = Variable(label.to(device)).squeeze()
                  optimizer.zero_grad()
                  pred = model(img)
                  loss = loss_fn(pred, label)
                  loss.backward()
                  optimizer.step()
                  val_loss.append(loss.item())
                  num_target += len(label)
                  correct += (pred.argmax(dim=1) == label).sum().item()
                  tepoch.set_postfix(loss=loss.item(), val_accuracy=correct / num_target)
        if i % 5 == 4 and i != epoch - 1:
            if valid:
                torch.save(model.state_dict(), 'new_r90_eff_final_imgsz%d_e%d.pt' % (img_size, i + 1))
            else:
                torch.save(model.state_dict(), 'new_r90_eff_final_no_val_imgsz%d_e%d.pt' % (img_size, i + 1))
    '''
    plt.plot(training_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.show()
    '''
    if valid:
        torch.save(model.state_dict(), 'new_r90_eff_final_imgsz%d.pt' % img_size)
    else:
        torch.save(model.state_dict(), 'new_r90_eff_final_no_val_imgsz%d.pt' % img_size)
