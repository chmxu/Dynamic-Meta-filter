from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        input_ = inputs
        input_ = input_.view(input_.size(0), input_.size(1), -1)

        log_probs = self.logsoftmax(input_)
        targets_ = torch.zeros(input_.size(0), input_.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets_ = targets_.unsqueeze(-1)
        targets_ = targets_.cuda()
        loss = (- targets_ * log_probs).mean(0).sum() 
        return loss / input_.size(2)
