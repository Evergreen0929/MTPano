# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

from shutil import ignore_patterns
import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio

class DepthMeter(object):
    def __init__(self, database, ignore_index=255, min_depth=1e-3, max_depth=None):
        self.database = database
        self.ignore_index = ignore_index
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.abs_rel = 0.0
        self.sq_rel = 0.0
        
        # Accuracy metrics accumulators
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        
        self.n_valid = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        
        mask = (gt != self.ignore_index).bool()

        if self.min_depth is not None:
            mask = mask & (gt > self.min_depth)
        if self.max_depth is not None:
            mask = mask & (gt < self.max_depth)

        if mask.sum() == 0:
            return

        gt_val = gt[mask]
        pred_val = pred[mask]
        
        pred_val = torch.clamp(pred_val, min=1e-3)
        gt_val = torch.clamp(gt_val, min=1e-3)
        if self.max_depth is not None:
            pred_val = torch.clamp(pred_val, max=self.max_depth)
            gt_val = torch.clamp(gt_val, max=self.max_depth)

        n_valid_sample = mask.float().sum().item()
        self.n_valid += n_valid_sample

        
        # RMSE
        rmse_tmp = torch.pow(gt_val - pred_val, 2)
        self.total_rmses += rmse_tmp.sum().item()

        # RMSE Log
        if self.database == 'Matterport3D':
            log_rmse_tmp = torch.pow(torch.log10(gt_val) - torch.log10(pred_val), 2)
        else:
            log_rmse_tmp = torch.pow(torch.log(gt_val) - torch.log(pred_val), 2)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        # Abs Rel
        self.abs_rel += (torch.abs(gt_val - pred_val) / gt_val).sum().item()

        # Sq Rel
        self.sq_rel += (((gt_val - pred_val) ** 2) / gt_val).sum().item()

        # Accuracy (delta metrics)
        # thresh = max(gt/pred, pred/gt)
        thresh = torch.max((gt_val / pred_val), (pred_val / gt_val))
        self.a1 += (thresh < 1.25).float().sum().item()
        self.a2 += (thresh < 1.25 ** 2).float().sum().item()
        self.a3 += (thresh < 1.25 ** 3).float().sum().item()

    def get_score(self, verbose=True):
        eval_result = dict()
        
        if self.n_valid == 0:
            return {k: 0.0 for k in ['rmse', 'log_rmse', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']}
        
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        eval_result['abs_rel'] = self.abs_rel / self.n_valid
        eval_result['sq_rel'] = self.sq_rel / self.n_valid
        eval_result['a1'] = self.a1 / self.n_valid
        eval_result['a2'] = self.a2 / self.n_valid
        eval_result['a3'] = self.a3 / self.n_valid

        if verbose:
            print('Results for depth prediction')
            print('{:<15} {:.4f}'.format('RMSE', eval_result['rmse']))
            print('{:<15} {:.4f}'.format('Log RMSE', eval_result['log_rmse']))
            print('{:<15} {:.4f}'.format('Abs Rel', eval_result['abs_rel']))
            print('{:<15} {:.4f}'.format('Sq Rel', eval_result['sq_rel']))
            print('{:<15} {:.4f}'.format('Delta < 1.25', eval_result['a1']))
            print('{:<15} {:.4f}'.format('Delta < 1.25^2', eval_result['a2']))
            print('{:<15} {:.4f}'.format('Delta < 1.25^3', eval_result['a3']))

        return eval_result
