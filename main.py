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
num_hidden_units = 20

W = torch.randn((num_visible_units,
                 num_hidden_units))
W = W * 0.2
W = torch.tensor(W, requires_grad = True, device = 'cuda')
b_v = torch.zeros(num_visible_units, dtype = W.dtype,
                  device = W.device, requires_grad = True)
b_h = torch.zeros(num_hidden_units, dtype = W.dtype,
                  device = W.device, requires_grad = True)

num_particles = 200
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
 
optimizer = optim.Adam([W, b_v, b_h], lr = 0.001)

train_image = data['train_image']
test_image = data['test_image']
test_image = torch.tensor(test_image, dtype = W.dtype, device = W.device)
batch_size = num_particles

bias_vh = None
bias_v = None
bias_h = None

for i in range(20):
    print("batch {:>4d}".format(i))
    ## get a batch of data
    data_V = train_image[i*batch_size:(i+1)*batch_size, :]
    data_V = torch.tensor(data_V, dtype = W.dtype, device = W.device)
    random = torch.rand_like(data_V)
    data_V = (random <= data_V).float()
    
    ## calculate data expectation
    P = V2P(data_V)
    grad_W_data = torch.matmul(data_V.t(), P) / batch_size
    grad_b_v_data = torch.mean(data_V, 0)
    grad_b_h_data = torch.mean(P,0)

    ## updates particles with Gibbs sampling
    samples_V = []
    samples_H = []
    for i in range(100):
        V = H2V(H)
        H = V2H(V)
        if i % 20 == 0:
            samples_V.append(V)
            samples_H.append(H)
    samples_V = torch.cat(samples_V)
    samples_H = torch.cat(samples_H)
    num_samples = samples_V.shape[0]

    ## calculate model expectation for v_i*h_j
    energy_vh = calculate_energy_matrix_pair(W.detach(), b_v.detach(), b_h.detach(), samples_V, samples_H)
    energy_vh = energy_vh - energy_vh.min(-1, keepdim = True)[0]
    count_vh = calculate_states_count_pair(samples_V, samples_H)
    mask_vh = (count_vh != 0).float()
    count_vh = count_vh.float()
    F_vh, bias_vh = calculate_free_energy_mbar(energy_vh, count_vh, mask_vh, bias_vh)

    prob_vh = torch.exp(-F_vh)
    grad_W_model = prob_vh[:,:,3] / prob_vh.sum(-1)

    ## calculate model expectation for v_i/h_j
    energy_v, energy_h = calculate_energy_matrix_single(W.detach(), b_v.detach(), b_h.detach(), samples_V, samples_H)
    energy_v = energy_v - energy_v.min(-1, keepdim = True)[0]
    energy_h = energy_h - energy_h.min(-1, keepdim = True)[0]
    
    count_v, count_h = calculate_states_count_single(samples_V, samples_H)
    mask_v, mask_h = (count_v != 0).float(), (count_h != 0).float()
    count_v, count_h = count_v.float(), count_h.float()
    F_v, bias_v = calculate_free_energy_mbar(energy_v, count_v, mask_v, bias_v)
    F_h, bias_h = calculate_free_energy_mbar(energy_h, count_h, mask_h, bias_h)    

    prob_v, prob_h = torch.exp(-F_v), torch.exp(-F_h)
    grad_b_v_model,grad_b_h_model = prob_v[:,1]/prob_v.sum(-1), prob_h[:,1]/prob_h.sum(-1)

    ## combined data expectation and model expectation to calculate gradients
    W.grad = -(grad_W_data - grad_W_model)
    b_v.grad = -(grad_b_v_data - grad_b_v_model)
    b_h.grad = -(grad_b_h_data - grad_b_h_model)

    optimizer.step()

    logZ_v = calculate_log_partition_function_samples(W.detach(), b_v.detach(),
                                                      b_h.detach(), test_image)
    logZ = calculate_log_partition_function(W.detach(), b_v.detach(), b_h.detach())

    print("log probability of test image: {:>7.3f}".format(logZ_v - logZ))
