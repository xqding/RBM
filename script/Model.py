__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/11/07 17:37:59"

import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, num_visible_units, num_hidden_units):
        super(RBM, self).__init__()
        self.num_visible_units = num_visible_units
        self.num_hidden_units = num_hidden_units

        self.W = nn.Parameter(torch.randn((self.num_visible_units,
                                           self.num_hidden_units)))
        self.b_v = nn.Parameter(torch.zeros(self.num_visible_units))
        self.b_h = nn.Parameter(torch.zeros(self.num_hidden_units))        
        #self.register_parameter('W', self.W)

model = RBM(20,10)

    
