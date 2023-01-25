#!/usr/bin/python3
#########################################################################
# Author: Tina Issa
# Last updated: October 15, 2021
# This class defiines the objective functions of the neural network. 
#########################################################################

import torch
import torch.nn as nn

def l21_norm(W): 
    return torch.sum(torch.linalg.norm(W, dim=-1))

def get_group_regularization(weights):
    const_coeff = lambda W: torch.sqrt(torch.tensor(W.size(dim=-1), dtype=torch.float32))
    #return torch.sum(torch.tensor([torch.multiply(const_coeff(W), l21_norm(W)) for name, W in model.named_parameters() if 'bias' not in name]))
    return torch.sum(torch.tensor([torch.multiply(const_coeff(W), l21_norm(W)) for name, W in weights.items() if 'bias' not in name]))

def sparse_group_lasso(weights):
    grouplasso = get_group_regularization(weights)
    #l1 = torch.linalg.norm(torch.cat([torch.reshape(x[1] ,[-1]) for x in model.named_parameters()], dim=0))
    l1 = torch.linalg.norm(torch.cat([torch.reshape(x[1] ,[-1]) for x in weights.items()], dim=0))
    
    sparse_lasso = grouplasso + l1
    return sparse_lasso

'''
def f1_norm(feat, label, model):
    cross_entropy_norm = nn.CrossEntropyLoss()(label, model(feat).long())
    return cross_entropy_norm
'''

def f1_norm(feat, label, model, lossWeight):
    criterion = nn.BCEWithLogitsLoss(pos_weight = lossWeight)
    return criterion(label, model(feat))

def f2_norm(max_b, weights):
    s_g_l = sparse_group_lasso(weights)
    sparse_group_lasso_norm = s_g_l / max_b
    return sparse_group_lasso_norm