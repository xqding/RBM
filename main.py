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

num_visible_units = 40
num_hidden_units = 20

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
energy = calculate_energy_matrix(W, b_v, b_h, samples_v, samples_h)
energy = energy - energy.min(-1, keepdim = True)[0]
count = calculate_states_count(W, b_v, b_h, samples_v, samples_h)
mask = (count != 0).float()
count = count.float()
F = calculate_free_energy_mbar(energy, count, mask)

samples_v = samples_v.float()
samples_h = samples_h.float()
prob_sample = torch.matmul(samples_v.t(), samples_h).float() / samples_h.shape[0]
prob_cd_1 = torch.matmul(prob_v.view((-1,1)), h.view((1,-1)))
prob_mbar = calculate_model_expectation_mbar(W, b_v, b_h, samples_v, samples_h)

sys.exit()

diff_sample = torch.mean(torch.abs(prob_sample - prob)).item()
diff_cd_1 = torch.mean(torch.abs(prob_cd_1 - prob)).item()
diff_mbar = torch.mean(torch.abs(prob_mbar - prob)).item()

angle_sample = calculate_angle(prob_sample.view(-1), prob.view(-1))
angle_cd_1 = calculate_angle(prob_cd_1.view(-1), prob.view(-1))
angle_mbar = calculate_angle(prob_mbar.view(-1), prob.view(-1))

print("diff_sample: {:.3f}, diff_cd_1: {:.3f}, diff_mbar: {:.3f}".format(diff_sample, diff_cd_1, diff_mbar))
print("angle_sample: {:.3f}, angle_cd_1: {:.3f}, angle_mbar: {:.3f}".format(angle_sample, angle_cd_1, angle_mbar))


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


loss_model = mbar_loss(energy, count, mask)
optimizer = optim.LBFGS(loss_model.parameters(), max_iter = 10, tolerance_change=1e-5)
previous_loss = loss_model()
previous_loss.backward()
previous_loss = previous_loss.item()
grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()

print("start loss: {:>7.5f}, start grad: {:>7.5f}".format(previous_loss, grad_max)) 
for i in range(100):
    def closure():
        optimizer.zero_grad()
        loss = loss_model()
        loss.backward()    
        return loss
    optimizer.step(closure)
    loss = loss_model().item()
    grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()
    print("step: {:>4d}, loss:{:>7.5f}, grad: {:>7.5f}".format(i, loss, grad_max)) 
    if np.abs(loss-previous_loss) <= 1e-4 or grad_max <= 1e-4:
        break
    previous_loss = loss

bias = loss_model.bias.data
tmp = -torch.log(count/num_samples)*mask
tmp[torch.isnan(tmp)] = 0
F = tmp - bias

## normalize F
prob = torch.exp(-F) * mask
prob = prob / prob.sum(-1, keepdim = True)
F = - torch.log(prob)
bias = -torch.log(count/num_samples) - F
bias[torch.isnan(bias)] = 0

biased_energy = energy + bias
tmp = torch.sum(torch.exp(-biased_energy)*mask, -1, keepdim = True)
F = -torch.log(torch.mean(torch.exp(-energy)/tmp, 0))

sys.exit()

bias_init = np.random.randn(np.prod(list(count.shape)))
num_samples = samples_v.shape[0]
obj_np, grad_np = mbar_loss_grad_np(bias_init, energy, count, mask)
x, f, d = optimize.fmin_l_bfgs_b(mbar_loss_grad_np, bias_init, iprint = 1, args = (energy, count, mask))
