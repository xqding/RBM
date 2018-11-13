__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import itertools
import sys    
sys.path.append("/home/xqding/course/projectsOnGitHub/FastMBAR/FastMBAR")
from FastMBAR import *

def calculate_log_partition_function(W, b_v, b_h):
    num_hidden_units = len(b_h)
    num_visible_units = len(b_v)
    assert(tuple(W.shape) == (num_visible_units, num_hidden_units))

    ## enumerate possible values of hidden units
    h_values_cpu = [i for i in itertools.product(range(2), repeat = num_hidden_units)]

    num_batch = 10
    batch_size = len(h_values_cpu)//10 + 1
    logZ_h_list = []
    for i in range(num_batch):
        h_values = torch.tensor(h_values_cpu[i*batch_size:(i+1)*batch_size],
                                dtype = W.dtype, device = W.device)

        ## given each possible value of hidden unit, visible units are independent.
        ## therefore the partition function is equal to the product of individual
        ## partition function.
        energy_v = -(torch.matmul(h_values, W.t()) + b_v)
        logZ_h = torch.matmul(h_values, b_h) + torch.sum(torch.log(1 + torch.exp(-energy_v)), -1)
        logZ_h_list.append(logZ_h.cpu())
        
    ## make the operation stable by substrating the maximum value
    logZ_h = torch.cat(logZ_h_list)
    tmp = torch.max(logZ_h)
    logZ_h = logZ_h - tmp
    logZ = torch.log(torch.sum(torch.exp(logZ_h))) + tmp    
    return logZ

def calculate_log_partition_function_samples(W, b_v, b_h, samples_v):
    num_hidden_units = len(b_h)
    num_visible_units = len(b_v)
    assert(num_visible_units == W.shape[0])
    assert(num_hidden_units == W.shape[1])    
    num_samples = samples_v.shape[0]
    
    energy_h = -(torch.matmul(samples_v, W) + b_h)
    logZ_v = torch.matmul(samples_v, b_v) + torch.sum(torch.log(1 + torch.exp(-energy_h)), -1)

    return torch.mean(logZ_v).item()


def calculate_model_expectation_brute_force(W, b_v, b_h):
    num_visible_units = len(b_v)    
    num_hidden_units = len(b_h)

    logZ = calculate_log_partition_function(W, b_v, b_h)
    prob = torch.zeros((num_visible_units, num_hidden_units))
    
    for i in range(num_visible_units):
        for j in range(num_hidden_units):
            idx_v = list(range(num_visible_units))
            idx_v.pop(i)

            idx_h = list(range(num_hidden_units))
            idx_h.pop(j)

            sub_W = W[idx_v, :][:, idx_h]

            ## v_i = 1, h_j = 1
            new_b_v = b_v + W[:, j]
            new_b_v = new_b_v[idx_v]
            new_b_h = b_h + W[i, :]
            new_b_h = new_b_h[idx_h]            
            logZ11 = calculate_log_partition_function(sub_W, new_b_v, new_b_h)    
            logZ11 += b_v[i] + b_h[j] + W[i,j]
            prob[i,j] = (torch.exp(logZ11-logZ)).item()
            
    return prob

def calculate_states_count_single(samples_v, samples_h):
    count_v = []
    count_h = []
    for value in range(2):
        count_v.append(torch.sum(samples_v == value, 0))
        count_h.append(torch.sum(samples_h == value, 0))
    count_v = torch.stack(count_v, -1)
    count_h = torch.stack(count_h, -1)    
    return count_v, count_h

def calculate_energy_matrix_single(W, b_v, b_h, samples_v, samples_h):
    ## make sure dimensions of parameters agree with each other
    num_visible_units = len(b_v)
    num_hidden_units = len(b_h)
    assert(num_visible_units == W.shape[0])
    assert(num_hidden_units == W.shape[1])
    
    num_samples = samples_v.shape[0]
    assert(num_samples == samples_h.shape[0])
    assert(samples_v.shape[1] == num_visible_units)
    assert(samples_h.shape[1] == num_hidden_units)

    energy_matrix_v = -(torch.matmul(samples_h.float(), W.t()) + b_v)
    tmp = torch.zeros_like(energy_matrix_v)
    energy_matrix_v = torch.stack([tmp, energy_matrix_v], -1)

    energy_matrix_h = -(torch.matmul(samples_v.float(), W) + b_h)
    tmp = torch.zeros_like(energy_matrix_h)
    energy_matrix_h = torch.stack([tmp, energy_matrix_h], -1)

    return energy_matrix_v, energy_matrix_h
    
def calculate_states_count_pair(samples_v, samples_h):
    ## make sure dimensions of parameters agree with each other
    num_visible_units = samples_v.shape[1]
    num_hidden_units = samples_h.shape[1]
    num_samples = samples_v.shape[0]
    assert(num_samples == samples_h.shape[0])

    samples_v = samples_v.view(num_samples, num_visible_units, 1)
    samples_h = samples_h.view(num_samples, 1, num_hidden_units)

    count = []
    for value_v in range(2):
        tmp_v = samples_v == value_v        
        for value_h in range(2):
            tmp_h = samples_h == value_h
            tmp = tmp_v * tmp_h
            count.append(tmp)
    count = torch.stack(count, -1)
    count = count.sum(0)
    return count

def calculate_energy_matrix_pair(W, b_v, b_h, samples_v, samples_h):
    ## make sure dimensions of parameters agree with each other
    num_visible_units = len(b_v)
    num_hidden_units = len(b_h)
    assert(num_visible_units == W.shape[0])
    assert(num_hidden_units == W.shape[1])
    
    num_samples = samples_v.shape[0]
    assert(num_samples == samples_h.shape[0])
    assert(samples_v.shape[1] == num_visible_units)
    assert(samples_h.shape[1] == num_hidden_units)


    ## expand samples of visible units
    ## each sample of visible units is repeated by num_visible_units times.
    ## the i'th position of the i'th repeated sample is set to 0 or 1
    V = samples_v.repeat(1, num_visible_units).reshape(-1,num_visible_units,
                                                       num_visible_units)        
    V = V.float()
    idx_v = torch.arange(0, num_visible_units, dtype = torch.long)

    ## expand samples of hidden units (similar to samples of visible units)
    ## each sample of hidden units is repeated by num_hidden_units times.
    ## the i'th position of the i'th repeated sample is set to 0 or 1,
    H = samples_h.repeat(1, num_hidden_units).reshape(-1,num_hidden_units,
                                                       num_hidden_units)
    H = H.float()
    idx_h = torch.arange(0, num_hidden_units, dtype = torch.long)

    ## calculate energy matrix
    energy = []
    for value_v in range(2):
        for value_h in range(2):
            ## for each samples (v,h), set each pair of (v_i, h_j) to
            ## four possible states ((0,0),(0,1),(1,0),(1,1))
            ## and calculate energies
            V[:, idx_v, idx_v] = value_v
            H[:, idx_h, idx_h] = value_h
            vb = torch.matmul(V, b_v).unsqueeze(-1)
            hb = torch.matmul(H, b_h).unsqueeze(1)    
            vWh = torch.matmul(torch.matmul(V, W), H.transpose(1,2))    
            tmp_energy = - (vWh + vb + hb)
            energy.append(tmp_energy)
    energy = torch.stack(energy, -1)
    return energy

class mbar_loss(nn.Module):
    def __init__(self, energy, count, mask, bias = None):
        super(mbar_loss, self).__init__()
        self.energy = energy
        self.num_samples = energy.shape[0]
        self.count = count
        self.mask = mask

        if bias is None:
            bias  = torch.randn(count.size(), dtype = energy.dtype,
                               device = energy.device, requires_grad = True)
            bias.data = bias.data * self.mask
            self.bias = nn.Parameter(bias)            
        else:
            bias = torch.tensor(bias, device = energy.device, requires_grad = True)
            bias.data = bias.data * self.mask
            self.bias = nn.Parameter(bias)            
    
    def forward(self):
        tmp = self.energy + self.bias
        loss = 1.0 / (self.num_samples) * (torch.sum(torch.log(torch.sum(torch.exp(-tmp)*self.mask, -1))) + torch.sum(self.count*self.bias))
        return loss

def calculate_free_energy_mbar(energy, count, mask, bias = None, verbose = False):
    ## calculate bias energies for states with nonzero samples by
    ## optimizing the mbar loss functin with L-BFGS-B algorithm
    ## use the L-BFGS-B algorithm avaiable in pytorch.optim

    num_samples = energy.shape[0]
    
    ## mbar_loss object to calcualte loss and grad
    loss_model = mbar_loss(energy, count, mask, bias)
    optimizer = optim.LBFGS(loss_model.parameters(), max_iter = 10, tolerance_change=1e-4)

    ## calcualte loss and grad
    previous_loss = loss_model()
    previous_loss.backward()
    previous_loss = previous_loss.item()
    grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()

    ## minimize loss using L-BFGS-B
    if verbose:
        print("start loss: {:>7.5f}, start grad: {:>7.5f}".format(previous_loss, grad_max)) 
    for i in range(30):
        def closure():
            optimizer.zero_grad()
            loss = loss_model()
            loss.backward()    
            return loss
        optimizer.step(closure)
        loss = loss_model().item()
        grad_max = torch.max(torch.abs(loss_model.bias.grad)).item()
        
        if verbose:
            print("step: {:>4d}, loss:{:>7.5f}, grad: {:>7.5f}".format(i, loss, grad_max))
            
        ## stop criterion for L-BFGS-B
        ## this is added because the optim.LBFGS often returns nan values
        ## when it runs too many iterations.
        if np.abs(loss-previous_loss) <= 1e-3 or grad_max <= 1e-3:
            break
        previous_loss = loss

    ## using the bias energies to calculate free energies for states with
    ## nonzero samples
    bias = loss_model.bias.data
    tmp = -torch.log(count/num_samples)*mask
    tmp[torch.isnan(tmp)] = 0
    F = tmp - bias

    ## normalize free energyies of states with nonzero samples
    prob = torch.exp(-F) * mask
    prob = prob / prob.sum(-1, keepdim = True)
    F_nz = - torch.log(prob)

    ## update bias energy based on normalized F for states with
    ## nonzero samples
    bias = -torch.log(count/num_samples) - F_nz
    bias[torch.isnan(bias)] = 0

    ## calculate free energies for all states using
    ## importance sampling/Zwanzig equations
    ## note that F agrees with F_nz for states with nonzero samples
    biased_energy = energy + bias
    tmp = torch.sum(torch.exp(-biased_energy)*mask, -1, keepdim = True)
    F = -torch.log(torch.mean(torch.exp(-energy)/tmp, 0))

    bias[bias == float('inf')] = 0    
    return F, bias
    
def mbar_loss_grad(bias, energy, count, mask):
    tmp = energy + bias
    num_samples = energy.shape[0]
    loss = 1.0 / (num_samples) * (torch.sum(torch.log(torch.sum(torch.exp(-tmp)*mask, -1))) + torch.sum(count*bias))
    grad = 1.0 / (num_samples) * (-torch.sum((torch.exp(-tmp)*mask)/torch.sum(torch.exp(-tmp)*mask,-1,keepdim = True), 0) + count)
    return loss, grad

def mbar_loss_grad_np(bias, energy, count, mask):
    num_samples = energy.shape[0]
    
    bias = torch.tensor(bias, dtype = energy.dtype, device = energy.device)
    bias = bias.reshape(count.size())
    tmp = energy + bias

    # obj = 1.0 / (torch.sum(count)) * (torch.sum(torch.log(torch.sum(torch.exp(-tmp)*mask, -1))) + torch.sum(count * bias))
    # grad = 1.0 / (torch.sum(count)) * ( - torch.sum((torch.exp(-tmp)*mask)/torch.sum(torch.exp(-tmp)*mask, -1, keepdim = True), 0) + count)
    
    obj = 1.0 / (num_samples) * (torch.sum(torch.log(torch.sum(torch.exp(-tmp)*mask, -1))) + torch.sum(count * bias))
    grad = 1.0 / (num_samples) * ( - torch.sum((torch.exp(-tmp)*mask)/torch.sum(torch.exp(-tmp)*mask, -1, keepdim = True), 0) + count)

    obj = obj.cpu().item()
    grad = grad.cpu().numpy().reshape(-1)
    return obj, grad.astype(np.float64)    


def calculate_model_expectation_mbar(W, b_v, b_h, samples_v, samples_h):
    '''
    calcualte the model expection of <v_i*h_j> using MBAR approach
    '''

    ## make sure dimensions of parameters agree with each other
    num_visible_units = len(b_v)
    num_hidden_units = len(b_h)
    assert(num_visible_units == W.shape[0])
    assert(num_hidden_units == W.shape[1])
    
    num_samples = samples_v.shape[0]
    assert(num_samples == samples_h.shape[0])
    assert(samples_v.shape[1] == num_visible_units)
    assert(samples_h.shape[1] == num_hidden_units)

    ## calculate the model expection of <v_i*h_j> for each pair of (i,j)
    ## using MBAR
    prob_mbar = torch.zeros((num_visible_units, num_hidden_units))
    
    for i in range(num_visible_units):
        for j in range(num_hidden_units):
            tmp_samples_v = samples_v.clone().float()
            tmp_samples_h = samples_h.clone().float()

            ## for each sample, calculate its energy for four different states:
            ## 1. set v_i = 0, v_j = 0, calculate energy_00
            ## 2. set v_i = 0, v_j = 1, calculate energy_01
            ## 3. set v_i = 1, v_j = 0, calculate energy_10
            ## 4. set v_i = 1, v_j = 1, calculate energy_11
            
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

            ## combined energy_00, energy_01, energy_10, and energy_11 into
            ## an energy matrix used by MBAR
            energy = torch.stack((energy_00, energy_01, energy_10, energy_11))

            ## count number of samples for eacho of the four different states
            ## 1. (v_i, h_j) = (0, 0)
            ## 2. (v_i, h_j) = (0, 1)
            ## 3. (v_i, h_j) = (1, 0)
            ## 4. (v_i, h_j) = (1, 1)            
            samples = torch.stack((samples_v[:,i], samples_h[:,j])).t()
            samples = samples.tolist()
            count = (samples.count([0,0]), samples.count([0,1]),
                     samples.count([1,0]), samples.count([1,1]))

            ## calculate free energies (negative log of partitioin function) of
            ## the four states using MBAR
            mbar = FastMBAR(energy.cpu().numpy(), np.array(count), cuda = True)
            F, _ = mbar.calculate_free_energies(bootstrap = False)
            F = torch.from_numpy(F)
            
            ## convert free energyies into probability
            Z = torch.exp(-F)
            prob_mbar[i,j] = Z[-1] / torch.sum(Z)
            
    return prob_mbar

def calculate_angle(v1, v2):
    v1_norm = torch.sqrt(torch.sum(v1**2))
    v2_norm = torch.sqrt(torch.sum(v2**2))    
    innner_product = torch.sum(v1 * v2)
    angle = torch.acos(innner_product / (v1_norm * v2_norm))
    return angle / math.pi * 180
