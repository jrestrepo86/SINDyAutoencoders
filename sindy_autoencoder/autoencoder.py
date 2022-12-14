#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy as sp
import sklearn as sc
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, plot
from torch.nn import functional as F


# ------------------------------------------------------------------------
# Funciones de activación
def get_activation_fn(afn):
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "selu": nn.SELU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        'None': None
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


# ------------------------------------------------------------------------
# Funciones de costo
def get_loss_fn(lfn, **kwargs):
    loss_functions = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        'l1-mse': L1MSE(kwargs.get('a', None)),
    }

    if lfn not in loss_functions:
        raise ValueError(
            f"'{lfn}' is not included in activation_functions. Use below one \n {loss_functions.keys()}"
        )

    return loss_functions[lfn]


class L1MSE(nn.Module):

    def __init__(self, a=1):
        super(L1MSE, self).__init__()
        if a is None:
            self.a = 1
        else:
            self.a = a

    def forward(self, output, input):
        x = output - input
        loss = self.a * nn.MSELoss(x) + (1 - self.a) * nn.L1Loss(x)
        return loss.mean()


# ------------------------------------------------------------------------
# Modelo AutoEncoder
class AutoEncoder(nn.Module):

    def __init__(self,
                 encoder_sizes=[12, 10, 8, 6, 4],
                 encoder_afn='prelu',
                 decoder_afn='prelu',
                 device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = encoder_sizes[::-1]
        self.encoder_afn = encoder_afn
        self.decoder_afn = decoder_afn

        # build encoder / decoder
        if encoder_afn == 'prelu':
            encoder = PReLU_Layers(self.encoder_sizes)
        else:
            encoder = simpleLayers(self.encoder_sizes, encoder_afn)
        if decoder_afn == 'prelu':
            decoder = PReLU_Layers(self.decoder_sizes)
        else:
            decoder = simpleLayers(self.decoder_sizes, decoder_afn)

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, input):
        return self.decoder.forward(self.encoder.forward(input))


# ------------------------------------------------------------------------
# Capas
class simpleLayers(nn.Module):

    def __init__(self, sizes, afn):
        super().__init__()
        activation_fn = get_activation_fn(afn)
        layers = []
        for m, n in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features=m, out_features=n, bias=True))
            layers.append(activation_fn())
        layers.pop(-1)
        self.layers = nn.Sequential(*layers)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            nn.init.constant_(layer.bias, 0)

    def forward(self, input):
        return self.layers(input)


class PReLU_Layers(nn.Module):

    def __init__(self, sizes):
        super().__init__()
        activation_fn = get_activation_fn('prelu')
        layers = []
        for m, n in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features=m, out_features=n, bias=True))
            layers.append(activation_fn(num_parameters=n))
        layers.pop(-1)
        self.layers = nn.Sequential(*layers)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            nn.init.constant_(layer.bias, 0)

    def forward(self, input):
        return self.layers(input)


# ------------------------------------------------------------------------
# Entrenamiento
def train(loader, model, optimizer, loss_fn, n_epoch, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device
    torch.device(device)

    model.train()
    Losses = []
    for epoch in range(n_epoch):
        loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            train_loss = loss_fn(output, batch)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        Losses.append(loss / len(loader))
        print(f'epoch : {epoch+1}/{n_epoch}, loss = {Losses[-1]:.6f}')
    return Losses


def test(loader, model, optimizer, loss, n_epoch, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device
    torch.device(device)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            test_loss = loss(batch, output)
            loss += test_loss.item()
    return loss
