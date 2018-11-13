__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.optimize as optimize
import itertools
import sys
from function import *

sys.path.append("/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR")
from FastMBAR import *

num_visible_units = 26
num_hidden_units = 24

#torch.random.manual_seed(0)
W = torch.randn((num_visible_units,
                 num_hidden_units))
b_v = torch.randn(num_visible_units)
b_h = torch.randn(num_hidden_units)

W = W.cuda()
b_v = b_v.cuda()
b_h = b_h.cuda()

#prob = calculate_model_expectation_brute_force(W, b_v, b_h)
v = torch.randint_like(b_v, 0, 2)
burn_in_steps = 1000
for i in range(burn_in_steps):
    energy_h = -(torch.matmul(W.t(), v) + b_h)
    prob_h = 1 / (1 + torch.exp(energy_h))
    random_u = torch.rand_like(prob_h)
    h = (random_u <= prob_h).float()

    energy_v = -(torch.matmul(W, h) + b_v)
    prob_v = 1 / (1 + torch.exp(energy_v))
    random_u = torch.rand_like(prob_v)
    v = (random_u <= prob_v).float()

num_steps = 200
samples_v = []
samples_h = []
for i in range(num_steps):
    energy_h = -(torch.matmul(W.t(), v) + b_h)
    prob_h = 1 / (1 + torch.exp(energy_h))
    random_u = torch.rand_like(prob_h)
    h = (random_u <= prob_h).float()

    energy_v = -(torch.matmul(W, h) + b_v)
    prob_v = 1 / (1 + torch.exp(energy_v))
    random_u = torch.rand_like(prob_v)
    v = (random_u <= prob_v).float()

    if i % 2 == 0:
        samples_h.append(h.int())
        samples_v.append(v.int())

samples_h = torch.stack(samples_h)
samples_v = torch.stack(samples_v)
num_samples = samples_v.shape[0]
print("calculate energy ...")
energy_vh = calculate_energy_matrix_pair(W, b_v, b_h, samples_v, samples_h)
energy_vh = energy_vh - energy_vh.min(-1, keepdim = True)[0]
count_vh = calculate_states_count_pair(samples_v, samples_h)
mask_vh = (count_vh != 0).float()
count_vh = count_vh.float()
print("calculate model expectation")
F_vh,bias_vh = calculate_free_energy_mbar(energy_vh, count_vh, mask_vh)

count_v, count_h = calculate_states_count_single(samples_v, samples_h)
energy_v, energy_h = calculate_energy_matrix_single(W, b_v, b_h, samples_v, samples_h)
mask_v = (count_v != 0).float()
count_v = count_v.float()
F_v, bias_v = calculate_free_energy_mbar(energy_v, count_v, mask_v)
mask_h = (count_h != 0).float()
count_h = count_h.float()
F_h, bias_h = calculate_free_energy_mbar(energy_h, count_h, mask_h)

samples_v = samples_v.float()
samples_h = samples_h.float()
prob_sample = torch.matmul(samples_v.t(), samples_h).float() / samples_h.shape[0]
prob_cd_1 = torch.matmul(prob_v.view((-1,1)), h.view((1,-1)))
prob_mbar = calculate_model_expectation_mbar(W, b_v, b_h, samples_v, samples_h)


diff_sample = torch.mean(torch.abs(prob_sample - prob)).item()
diff_cd_1 = torch.mean(torch.abs(prob_cd_1 - prob)).item()
diff_mbar = torch.mean(torch.abs(prob_mbar - prob)).item()

angle_sample = calculate_angle(prob_sample.view(-1), prob.view(-1))
angle_cd_1 = calculate_angle(prob_cd_1.view(-1), prob.view(-1))
angle_mbar = calculate_angle(prob_mbar.view(-1), prob.view(-1))

print("diff_sample: {:.3f}, diff_cd_1: {:.3f}, diff_mbar: {:.3f}".format(diff_sample, diff_cd_1, diff_mbar))
print("angle_sample: {:.3f}, angle_cd_1: {:.3f}, angle_mbar: {:.3f}".format(angle_sample, angle_cd_1, angle_mbar))


sys.exit()
