__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import sys    

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

def calculate_model_expection(W, b_v, b_h):
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
