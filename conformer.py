"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


# Convolution module.
# This class extracts local features from EEG signals using convolutional layers
# and pooling operations.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        num_channels = 64

        # Small convolutional neural network
        self.shallownet = nn.Sequential(
            # Two Conv2d layers to extract local features from EEG signals.
            # First apply a 1x25 convolution (temporal direction) to each channel of the input.
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            # Then apply a 64x1 convolution, combining features from different channels.
            nn.Conv2d(40, 40, (64, 1), (1, 1)),
            # BatchNorm2d layer to standardize the outpots of the convolutional layers.
            nn.BatchNorm2d(40),
            # ELU activation function to introduce non-linearity.
            nn.ELU(),
            # Average pooling to reduce the dimensionality of the feature maps.
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            # Dropout layer to prevent overfitting.
            nn.Dropout(0.5),
        )

        # Linear projection onto lower-dimensional embedding space.
        self.projection = nn.Sequential(
            # Conv2d layer for linear projection.
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            # Rearrange layer from (batch_size, channels, height, width) to 
            # (batch_size, height * width, channels) for the multi-head attention layer.
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    # Forward pass of the module.
    # Expected input shape: (batch_size, channels, height, width).
    # Width is the temporal dimension, i.e. the number of time steps.
    # Output: (batch_size, height * width, emb_size) where height * width is the number of patches.
    # Patches are slices of the input along the time dimension.
    def forward(self, x: Tensor) -> Tensor:
        print("Shape of input to PatchEmbedding: ", x.shape)
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        
        # The dimensions of the linear layers in this sequence will be determined later
        self.fc = None 

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        
        # If self.fc has not been defined yet, or if the input size has changed,
        # define it now with the correct input size
        if self.fc is None or self.fc[0].in_features != x.size(-1):
            self.fc = nn.Sequential(
                nn.Linear(x.size(-1), 256),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2)
            )
            self.fc = self.fc.to(x.device)  # Make sure self.fc is on the right device
            
        out = self.fc(x.float())
        return x, out



class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub, total_data, test_tmp):
        super(ExP, self).__init__()
        self.batch_size = 8 # 72
        self.n_epochs = 2000
        self.c_dim = 2
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '../data/'

        self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.HalfTensor
        # self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        # summary(self.model, (1, 64, 1000))

        self.total_data = total_data
        self.test_tmp = test_tmp

        # Use half FP precision in order to save memory
        # self.model = self.model.half()


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(-1, 1):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 2), 1, 64, 256))
            for ri in range(int(self.batch_size / 2)):
                for rj in range(8):
                    if (tmp_data.shape[0] == 0):
                        print("tmp_data.shape[0] == 0")
                        break
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 32:(rj + 1) * 32] = tmp_data[rand_idx[rj], :, :,
                                                                    rj * 32:(rj + 1) * 32]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 2)])
            appended = tmp_label[:int(self.batch_size / 2)]
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()

        return aug_data, aug_label

    def get_source_data(self):
        # train data
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        # Reduce FP precision of Tensor to reduce memory consumption
        self.train_data = self.train_data

        self.train_data = np.transpose(self.train_data, (2, 0, 1))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = np.array(self.train_label)

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_data = self.test_tmp['data']
        self.test_data = self.test_data
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 0, 1))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = np.array(self.test_label)


        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        print("Data loaded")

        test_data = torch.from_numpy(test_data).type(self.Tensor)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        print("Test data loaded")
        print("Test data shape: ", test_data.shape)
        print("Test label shape: ", test_label.shape)
        print("Test label: ", test_label)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        print("Optimizer loaded")

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        print("Test data and label loaded")

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            torch.cuda.empty_cache()
            print("Epoch: ", e)
            # if /home/s1824086/data/hello.txt exists, write to it
            if os.path.exists('/home/s1824086/data/hello.txt'):
                with open('/home/s1824086/data/hello.txt', 'w') as f:
                    f.write("Epoch: " + str(e) + "\n")
            # in_epoch = time.time()
            self.model.train()
            scaler = torch.cuda.amp.GradScaler()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # print("aug_data", aug_data)
                img_combined = torch.cat((img, aug_data)).type(self.Tensor)
                label_combined = torch.cat((label, aug_label))

                del img, label, aug_data, aug_label

                # forward pass
                with torch.cuda.amp.autocast():
                    tok, outputs = self.model(img_combined)
                    loss = self.criterion_cls(outputs, label_combined)
                    # print("Outputs: ", outputs)
                    # print("Label: ", label)
                    # print("Loss: ", loss)

                self.optimizer.zero_grad()

                # During this step, values become nan using FP16...
                # loss.backward()
                # Instead try backward pass with scale
                scaler.scale(loss).backward()

                # optimizer step
                scaler.step(self.optimizer)

                scaler.update()

                # Delete tensors that are no longer needed
                del img_combined, tok


            # out_epoch = time.time()

            print("Now testing...")
            torch.cuda.empty_cache()
            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                with torch.cuda.amp.autocast():
                    Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label_combined).cpu().numpy().astype(int).sum()) / float(label_combined.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred


        torch.save(self.model.module.state_dict(), 'model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
