#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.append('../sindy_autoencoder/')
import autoencoder as au

# cargar datos
DATA_PATH = '../data'
data = sp.io.loadmat(f'{DATA_PATH}/AB06_sindy.mat')['angleData']
data = data.astype(np.float32)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
train_data, val_test_data = train_test_split(scaled_data, test_size=0.3)
test_data, val_data = train_test_split(val_test_data, test_size=0.5)

# Modelo
model_param = {
    'layers': [12, 10, 8, 6],
    'encoder_afn': 'prelu',
    'decoder_afn': 'prelu',
    'loss_fn': 'mse',
}
trainig_params = {
    'lr': 0.025,
    'n_epoch': 300,
    'batch_size': 250,
}

model = au.AutoEncoder(
    encoder_sizes=model_param['layers'],
    encoder_afn=model_param['encoder_afn'],
    decoder_afn=model_param['decoder_afn'],
    device=None,
)
optimizer = torch.optim.Adam(model.parameters(), lr=trainig_params['lr'])
loss_fn = au.get_loss_fn('mse')
train_loader = DataLoader(train_data,
                          batch_size=trainig_params['batch_size'],
                          shuffle=True,
                          num_workers=4)
test_loader = DataLoader(test_data,
                         batch_size=32,
                         shuffle=False,
                         num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

# Training
Losses = au.train(train_loader,
                  model,
                  optimizer,
                  loss_fn,
                  n_epoch=trainig_params['n_epoch'])

plt.plot(Losses)
plt.show()
