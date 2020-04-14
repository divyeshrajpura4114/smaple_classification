#!/usr/bin/env python

import os
import time
import glob
import h5py
import numpy as np
from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


class StatsPool(nn.Module):
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        x = x.transpose(1,2)
        means = torch.mean(x, dim=1)
        stds = torch.std(x,dim=1)
        x = torch.cat([means, stds], dim=1)
        return x

class SampleModel(nn.Module):
    def __init__(self, ):
        super(SampleModel, self).__init__()
        self.model = nn.Sequential()
        self.in_dim = 20
        self.conv_output_size = [512,512,512,512,1500]
        self.conv_kernel_size = [5,3,3,1,1]
        self.conv_dilation_size = [1,2,3,1,1]
        self.droput_p = 0.3
        for index in range(5):
                self.model.add_module("tdnn{0}".format(index),nn.Sequential(
                                nn.Conv1d(in_channels = self.in_dim, 
                                            out_channels = self.conv_output_size[index],
                                            kernel_size = self.conv_kernel_size[index], 
                                            dilation = self.conv_dilation_size[index]),
                                nn.ReLU(),
                                nn.BatchNorm1d(self.conv_output_size[index]),
                                #nn.Dropout(p = self.droput_p)
                                ))
                self.in_dim = self.conv_output_size[index]

        self.model.add_module("stats_pool", StatsPool())

        self.linear_output_size = [512,512,1739]
        self.in_dim = 3000
        for index in range(2):
                self.model.add_module("linear{0}".format(index), nn.Linear(in_features = self.in_dim, 
                                            out_features = self.linear_output_size[index]))
                self.model.add_module("relu{0}".format(index), nn.ReLU())
                self.in_dim = self.linear_output_size[index]
                
        self.model.add_module("linear{0}".format(3), nn.Linear(in_features = self.in_dim, 
                                            out_features = self.linear_output_size[-1]))
        
        self.model.apply(self.init_weights)

    def forward(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12, out=None)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

class SampleDataset(Dataset):
    def __init__(self, feat_dir, feat_file):
        self.feature_path = os.path.join(feat_dir, feat_file)
        with h5py.File(self.feature_path, 'r') as f:
            self.file_list = list(f.keys())
       
    def __getitem__(self, index):
        utt_id = self.file_list[index]
        utt_data = {}
        with h5py.File(self.feature_path, 'r') as f:
            utt_data = f[utt_id]
            return utt_data['feature'][()], utt_data['label'][()]

    def __len__(self):
        return len(self.file_list)

def train_epoch(train_loader, epoch):
    model.train().to(device)
    global iteration
    
    train_loss = 0

    for batch_id, (batch_feature, batch_label) in enumerate(train_loader):
        batch_feature = batch_feature.contiguous().to(device)
        batch_label = batch_label.to(device)
        
        batch_output = model(batch_feature.transpose(1,2))
        
        loss = model_loss(batch_output, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.data.item()
    
        mesg = "Epoch : {0}[{1}/{2}] | Iteration : {3} | Loss : {4:.4f} | Total Loss : {5:.4f} \n".format(
                    epoch+1, batch_id+1, len(train_loader), 
                    iteration, loss.data.item(), train_loss/(batch_id + 1))
        print(mesg)
        iteration = iteration + 1
    
    train_loss = train_loss/len(train_loader)
    print('-' * 85)
    print('Epoch {0:3d} | Training Loss {1:.4f}'.format(epoch+1, train_loss))
    return train_loss
    
def do_training():
    trainLoss = []
    start_epoch = 0
    end_epoch = 10

    for epoch in range(start_epoch, end_epoch):
        
        start_time = datetime.now()

        trainLoss.append(train_epoch(train_loader, epoch))

        model.eval().cpu()

        end_time = datetime.now()
        print('Start Time : {0} | Elapsed Time: {1}'.format(str(start_time), str(end_time - start_time)))
        print('-' * 85)
        print('\n')

        # if scheduler.get_last_lr()[0] > 0.00015:
        scheduler.step()

if __name__ == '__main__':
    # Its sample program for multi-class classification.

    # Keep this file (train_dnn_sample.py) and features folder in root directory and set root_dir with it.
    root_dir = '/home/speechlab/Divyesh/sample'
    feat_dir = os.path.join(root_dir, 'features')
    feat_file = 'features.hdf5'

    device = 'cuda'

    iteration = 1

    model = SampleModel().to(device)
    model_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.75)

    train_data = SampleDataset(feat_dir, feat_file)
    train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)

    do_training()


    # Please ignore below code, it is to generate sample.hdf5 file

    # file_list = glob.glob('/home/speechlab/Divyesh/divyesh/sdsv/data/val/batches_random_64/*')
    # feature_path = '/home/speechlab/Divyesh/divyesh/sdsv/features/mfcc/val'
    # result_path = '/home/speechlab/Divyesh/divyesh/sdsv/features/mfcc/sample'
    # with h5py.File(os.path.join(result_path, 'sample.hdf5'),'a') as f:
    #     for index in range(0,10):
    #         with open(file_list[index],'r') as f1:
    #             utt_list = f1.read().splitlines()
            
    #         for utt_data in utt_list:
    #             utt_id, _, utt_label = utt_data.split()
    #             with h5py.File(os.path.join(feature_path, utt_id + '.hdf5'), 'r') as f2:
    #                 utt_feature = np.array(f2['features'][:][0:200,0:20])
    #                 grp = f.create_group(utt_id)                
    #                 grp.create_dataset("feature", data = utt_feature)
    #                 grp.create_dataset("label", data = int(utt_label))