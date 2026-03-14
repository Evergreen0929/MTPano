#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.loss_functions import L1Loss

class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.aux_weight = 0.3
        self.ext_weight = 0.003
        
        self.ext_tasks = ['grad', 'sdf', 'point']
        self.ext_loss_ft = {t: L1Loss() for t in self.ext_tasks}

        self.max_depth_loss = 10.0 if p.train_db_name != 'Deep360' else 20.0
    
    def forward(self, pred, gt, tasks, ext_gt=None, clamp_depth=True):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        if clamp_depth and 'depth' in out.keys():
            out['depth'] = out['depth'].clamp(max=self.max_depth_loss)

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        if self.p.intermediate_supervision:
            aux_preds = pred['aux_preds']
            
            for task in tasks:
                aux_loss = self.loss_ft[task](aux_preds[task], gt[task])
                if clamp_depth and task == 'depth':
                    aux_loss = aux_loss.clamp(max=self.max_depth_loss)
                out[f'aux_{task}'] = aux_loss
                out['total'] += self.loss_weights[task] * aux_loss * self.aux_weight
            
            if ext_gt is not None:
                ext_preds = pred['ext_preds']
                for task in ext_gt.keys():
                    ext_loss = self.ext_loss_ft[task](ext_preds[task], ext_gt[task])
                    if clamp_depth and task == 'point':
                        ext_loss = ext_loss.clamp(max=self.max_depth_loss)
                    out[f'ext_{task}'] = ext_loss
                    out['total'] += ext_loss * self.ext_weight

        return out


class SSLLoss(nn.Module):
    def __init__(self, p, loss_fn):
        super(SSLLoss, self).__init__()
        self.p = p
        self.loss_fn = loss_fn
    
    def forward(self, pred_feat_list, pseudo_feat_list):
        out = {level: self.loss_fn(pred_feat_list[level], pseudo_feat_list[level]) for level in range(len(pred_feat_list))}
        out['total'] = torch.sum(torch.stack([out[level] for level in range(len(pred_feat_list))]))
        return out
