__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import sys

from function import *

num_visible_units = 10
num_hidden_units = 10

torch.random.manual_seed(0)
W = torch.randn((num_visible_units,
                 num_hidden_units))
b_v = torch.randn(num_visible_units)
b_h = torch.randn(num_hidden_units)

prob = calculate_model_expection(W, b_v, b_h)

v = torch.randint(0, 2, (num_visible_units, ))
burn_in_steps = 10000
for i in range(burn_in_steps):
    energy_h = -(torch.matmul(W.t(), v) + b_h)
    prob_h = 1 / (1 + torch.exp(energy_h))
    random_u = torch.rand(num_hidden_units)
    h = (random_u <= prob_h).float()

    energy_v = -(torch.matmul(W, h) + b_v)
    prob_v = 1 / (1 + torch.exp(energy_v))
    random_u = torch.rand(num_visible_units)
    v = (random_u <= prob_v).float()
        
num_steps = 5
samples_v = []
samples_h = []
for i in range(num_steps):
    energy_h = -(torch.matmul(W.t(), v) + b_h)
    prob_h = 1 / (1 + torch.exp(energy_h))
    random_u = torch.rand(num_hidden_units)
    h = (random_u <= prob_h).float()

    energy_v = -(torch.matmul(W, h) + b_v)
    prob_v = 1 / (1 + torch.exp(energy_v))
    random_u = torch.rand(num_visible_units)
    v = (random_u <= prob_v).float()

    samples_h.append(h.int())
    samples_v.append(v.int())

samples_h = torch.stack(samples_h)
samples_v = torch.stack(samples_v)
prob_hat = torch.matmul(samples_v.t(), samples_h).float() / num_steps

sys.exit()
