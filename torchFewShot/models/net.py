from __future__ import absolute_import
from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchFewShot.models.resnet_drop import resnet12

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch import nn
from torch.nn import functional as F

from dconv.layers import DeformConv
from torchdiffeq import odeint as odeint

class DynamicWeights_(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1):
        super(DynamicWeights_, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

        padding = 1 if kernel == 3 else 0
        offset_groups = 1
        self.off_conv = nn.Conv2d(channels*2, 3*3*2, 5, 
                padding=2, dilation=dilation, bias=False)
        self.kernel_conv = DeformConv(channels, groups * kernel * kernel, 
                    kernel_size=3, padding=dilation, dilation=dilation, bias=False)

        self.K = kernel * kernel
        self.group = groups

    def forward(self, support, query):
        N, C, H, W = support.size()
        R = C // self.group
        offset = self.off_conv(torch.cat([query, support], 1))
        dynamic_filter = self.kernel_conv(support, offset)
        dynamic_filter = F.sigmoid(dynamic_filter)
        return dynamic_filter

class DynamicWeights(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1, nFeat=640):
        super(DynamicWeights, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        padding = 1 if kernel == 3 else 0
        offset_groups = 1
        self.unfold = nn.Unfold(kernel_size=(kernel, kernel), 
                                padding=padding, dilation=1)

        self.K = kernel * kernel
        self.group = groups
        self.nFeat = nFeat

    def forward(self, t=None, x=None):
        query, dynamic_filter = x
        N, C, H, W = query.size()
        N_, C_, H_, W_ = dynamic_filter.size()
        R = C // self.group
        dynamic_filter = dynamic_filter.reshape(-1, self.K)
            
        xd_unfold = self.unfold(query)

        xd_unfold = xd_unfold.view(N, C, self.K, H * W)
        xd_unfold = xd_unfold.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1, 3, 2, 4).contiguous().view(N * self.group * H * W, R, self.K)
        out1 = torch.bmm(xd_unfold, dynamic_filter.unsqueeze(2)) 
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(N, self.group * R, H, W)

        out1 = F.relu(out1)
        return (out1, torch.zeros([N_, C_, H_, W_]).cuda())

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x[0])
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-2, atol=1e-2, method='rk4')
        return out[0][1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Model(nn.Module):
    def __init__(self, num_classes=64, kernel=3, groups=1):
        super(Model, self).__init__()
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.global_clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

        self.dw_gen = DynamicWeights_(self.nFeat, 1, kernel, groups)
        self.dw = self.dw = ODEBlock(DynamicWeights(self.nFeat, 1, kernel, groups, self.nFeat))

    def reshape(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.unsqueeze(2).repeat(1, 1, n2, 1, 1, 1)
        ftest = ftest.unsqueeze(1).repeat(1, n1, 1, 1, 1, 1)
        return ftrain, ftest

    def process_feature(self, f, ytrain, num_train, num_test, batch_size):
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest = self.reshape(ftrain, ftest)

        # b, n2, n1, c, h, w
        ftrain = ftrain.transpose(1, 2)
        ftest = ftest.transpose(1, 2)
        return ftrain, ftest

    def get_score(self, ftrain, ftest, num_train, num_test, batch_size):
        b, n2, n1, c, h, w = ftrain.shape

        ftrain_ = ftrain.clone()
        ftest_ = ftest.clone()
        ftrain_ = ftrain_.view(-1, *ftrain.size()[3:])
        ftest_ = ftest_.view(-1, *ftest.size()[3:])

        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.reshape(-1, *ftrain_norm.size()[3:])
        conv_weight = ftrain_norm.mean(-1, keepdim=True).mean(-2, keepdim=True)

        filter_weight = self.dw_gen(ftrain_, ftest_)
        cls_scores = self.dw(x=(ftest_, filter_weight))
        cls_scores = cls_scores.view(b*n2, n1, *cls_scores.size()[1:])
        cls_scores = cls_scores.view(1, -1, *cls_scores.size()[3:])
        cls_scores = F.conv2d(cls_scores, conv_weight, groups=b*n1*n2, padding=1)
        cls_scores = cls_scores.view(b*n2, n1, *cls_scores.size()[2:])
        return cls_scores


    def get_global_pred(self, ftest, ytest, num_test, batch_size, K):
        h = ftest.shape[-1]
        ftest_ = ftest.view(batch_size, num_test, K, -1)
        ftest_ = ftest_.transpose(2, 3) 
        ytest_ = ytest.unsqueeze(3) 
        ftest_ = torch.matmul(ftest_, ytest_) 
        ftest_ = ftest_.view(batch_size * num_test, -1, h, h)
        global_pred = self.global_clasifier(ftest_)
        return global_pred

    def get_test_score(self, score_list):
        return score_list.mean(-1).mean(-1)


    def forward(self, xtrain, xtest, ytrain, ytest, global_labels=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain, ftest = self.process_feature(f, ytrain, num_train, 
                                                num_test, batch_size)
        cls_scores = self.get_score(ftrain, ftest,
                                             num_train, num_test, batch_size)

        if not self.training:
            return self.get_test_score(cls_scores)

        global_pred = self.get_global_pred(ftest, ytest, num_test, batch_size, K)
        return global_pred, cls_scores
