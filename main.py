__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import sys
from function import *

sys.path.append("/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR")
from FastMBAR import *

num_visible_units = 60
num_hidden_units = 10

#torch.random.manual_seed(0)
W = torch.randn((num_visible_units,
                 num_hidden_units))
b_v = torch.randn(num_visible_units)
b_h = torch.randn(num_hidden_units)

prob = calculate_model_expectation_brute_force(W, b_v, b_h)

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
        
num_steps = 20
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

    if i % 2 == 0:
        samples_h.append(h.int())
        samples_v.append(v.int())

samples_h = torch.stack(samples_h)
samples_v = torch.stack(samples_v)
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

i = 5
j = 4

tmp_samples_v = samples_v.clone().float()
tmp_samples_h = samples_h.clone().float()

tmp_samples_v[:, i] = 0
tmp_samples_h[:, j] = 0
energy_00 = - (torch.sum(torch.matmul(tmp_samples_v, W)*tmp_samples_h, -1) + 
               torch.matmul(tmp_samples_v, b_v) +
               torch.matmul(tmp_samples_h, b_h))

tmp_samples_v[:, i] = 0
tmp_samples_h[:, j] = 1
energy_01 = - (torch.sum(torch.matmul(tmp_samples_v, W)*tmp_samples_h, -1) + 
               torch.matmul(tmp_samples_v, b_v) +
               torch.matmul(tmp_samples_h, b_h))

tmp_samples_v[:, i] = 1
tmp_samples_h[:, j] = 0
energy_10 = - (torch.sum(torch.matmul(tmp_samples_v, W)*tmp_samples_h, -1) + 
               torch.matmul(tmp_samples_v, b_v) +
               torch.matmul(tmp_samples_h, b_h))

tmp_samples_v[:, i] = 1
tmp_samples_h[:, j] = 1
energy_11 = - (torch.sum(torch.matmul(tmp_samples_v, W)*tmp_samples_h, -1) + 
               torch.matmul(tmp_samples_v, b_v) +
               torch.matmul(tmp_samples_h, b_h))

samples = torch.stack((samples_v[:,i], samples_h[:,j])).t()
samples = samples.tolist()
count = (samples.count([0,0]), samples.count([0,1]),
         samples.count([1,0]), samples.count([1,1]))

energy = torch.stack((energy_00, energy_01, energy_10, energy_11))
mbar = FastMBAR(energy.numpy(), np.array(count), cuda = False)
F, _ = mbar.calculate_free_energies(bootstrap = False)
Z = np.exp(-F)
prob_mbar_ij = Z[-1] / np.sum(Z)

sys.exit()
