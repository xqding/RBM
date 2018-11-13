__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/13 01:07:42"

import numpy as np
import scipy.optimize as optimize
import itertools
import sys
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from function import *

with open("./data/data.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)

num_visible_units = 784
num_hidden_units = 30

W = torch.randn((num_visible_units,
                 num_hidden_units), requires_grad = True, device = 'cuda')
b_v = torch.zeros(num_visible_units, dtype = W.dtype,
                  device = W.device, requires_grad = True)
b_h = torch.zeros(num_hidden_units, dtype = W.dtype,
                  device = W.device, requires_grad = True)

num_particles = 100
V = torch.randint(high = 2, size = (num_particles, num_visible_units),
                  dtype = b_v.dtype, device = b_v.device)

def V2H(V):
    energy = -(torch.matmul(V, W) + b_h)
    prob = 1.0 / (1 + torch.exp(energy))
    random_u = torch.rand_like(prob)
    H = (random_u <= prob).float()
    return H

def V2P(V):
    energy = -(torch.matmul(V, W) + b_h)
    prob = 1.0 / (1 + torch.exp(energy))
    return prob

def H2V(H):
    energy = -(torch.matmul(H, W.t()) + b_v)
    prob = 1.0 / (1 + torch.exp(energy))
    random_u = torch.rand_like(prob)
    V = (random_u <= prob).float()
    return V

burn_in_steps = 1000
for i in range(burn_in_steps):
    H = V2H(V)
    V = H2V(H)
 
optimizer = optim.Adam([W, b_v, b_h])

train_image = data['train_image']
batch_size = 100

for i in range(60):
    print("batch {:>4d}".format(i))
    ## get a batch of data
    data_V = train_image[i*batch_size:(i+1)*batch_size, :]
    data_V = torch.tensor(data_V, dtype = W.dtype, device = W.device)

    ## calculate data expectation
    P = V2P(data_V)
    grad_W_data = torch.matmul(data_V.t(), P) / batch_size
    grad_b_v_data = torch.mean(data_V, 0)
    grad_b_h_data = torch.mean(P,0)

    ## updates particles with Gibbs sampling
    samples_V = []
    samples_H = []
    for i in range(10):
        V = H2V(H)
        H = V2H(V)
        if i % 2 == 0:
            samples_V.append(V)
            samples_H.append(H)
    samples_V = torch.cat(samples_V)
    samples_H = torch.cat(samples_H)
    num_samples = samples_V.shape[0]

    ## calculate model expectation
    energy = calculate_energy_matrix(W.detach(), b_v.detach(), b_h.detach(), samples_V, samples_H)
    energy = energy - energy.min(-1, keepdim = True)[0]
    count = calculate_states_count(W.detach(), b_v.detach(), b_h.detach(), samples_V, samples_H)
    mask = (count != 0).float()
    count = count.float()
    F = calculate_free_energy_mbar(energy, count, mask)

    prob = torch.exp(-F)
    grad_W_model = prob[:,:,3] / prob.sum(-1)
    grad_b_v_model = samples_V.mean(0)
    grad_b_h_model = samples_H.mean(0)

    ## combined data expectation and model expectation to calculate gradients
    W.grad = -(grad_W_data - grad_W_model)
    b_v.grad = -(grad_b_v_data - grad_b_v_model)
    b_h.grad = -(grad_b_h_data - grad_b_h_model)

    optimizer.step()
