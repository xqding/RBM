__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    h_values = [i for i in itertools.product(range(2), repeat = num_hidden_units)]
    h_values = torch.FloatTensor(h_values)

    ## given each possible value of hidden unit, visible units are independent.
    ## therefore the partition function is equal to the product of individual
    ## partition function.
    energy_v = -(torch.matmul(h_values, W.t()) + b_v)
    logZ_h = torch.matmul(h_values, b_h) + torch.sum(torch.log(1 + torch.exp(-energy_v)), -1)

    ## make the operation stable by substrating the maximum value
    tmp = torch.max(logZ_h)
    logZ_h = logZ_h - tmp
    logZ = torch.log(torch.sum(torch.exp(logZ_h))) + tmp    
    return logZ

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
            
            # ## v_i = 0, h_j = 0
            # new_b_v = b_v + 0
            # new_b_v = new_b_v[idx_v]
            # new_b_h = b_h + 0
            # new_b_h = new_b_h[idx_h]        
            # logZ00 = calculate_log_partition_function(sub_W, new_b_v, new_b_h)

            # ## v_i = 0, h_j = 1
            # new_b_v = b_v + W[:, j]
            # new_b_v = new_b_v[idx_v]    
            # new_b_h = b_h + 0
            # new_b_h = new_b_h[idx_h]            
            # logZ01 = calculate_log_partition_function(sub_W, new_b_v, new_b_h)
            # logZ01 += b_h[j]

            # ## v_i = 1, h_j = 0
            # new_b_v = b_v + 0
            # new_b_v = new_b_v[idx_v]    
            # new_b_h = b_h + W[i, :]
            # new_b_h = new_b_h[idx_h]    
            # logZ10 = calculate_log_partition_function(sub_W, new_b_v, new_b_h)
            # logZ10 += b_v[i]


            # tmp_logZ = torch.tensor((logZ00, logZ01, logZ10, logZ11))
            # tmp_logZ_max = torch.max(tmp_logZ)
            # tmp_logZ = tmp_logZ - tmp_logZ_max
            # logZZ = torch.log(torch.sum(torch.exp(tmp_logZ))) + tmp_logZ_max
            # diff = (logZ - logZZ).item()
            # assert(diff <= 1e-4)
    
    return prob

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
            mbar = FastMBAR(energy.numpy(), np.array(count), cuda = False)
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
