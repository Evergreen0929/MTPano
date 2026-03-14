# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import numpy as np

def normalize_tensor_strict(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out

class NormalsMeter(object):
    def __init__(self, ignore_index=255, align_360mtl=False):
        self.ignore_index = ignore_index
        self.align_360mtl = align_360mtl
        self.reset()

    def reset(self):
        self.all_errors = [] 
        
        if self.align_360mtl:
            self.sum_mean = 0.0
            self.sum_median = 0.0
            self.sum_rmse = 0.0
            self.sum_a11 = 0.0
            self.sum_a22 = 0.0
            self.sum_a30 = 0.0
            self.count_batches = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        B = gt.shape[0]
        pred = pred.permute(0, 3, 1, 2) # [B, C, H, W]
        pred = 2 * pred / 255 - 1 # reverse post-processing
        
        valid_mask = (gt != self.ignore_index).all(dim=1)

        if not self.align_360mtl:
            pred = normalize_tensor_strict(pred, dim=1)
            gt = normalize_tensor_strict(gt, dim=1)

            deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
            deg_diff = torch.masked_select(deg_diff, valid_mask)
            self.all_errors.append(deg_diff.detach().cpu())
        else:
            # === Align with 360MTL ===
            dot_products = torch.sum(pred * gt, dim=1)

            dot_products = torch.clamp(dot_products, -1.0, 1.0)
            angle_radians = torch.acos(dot_products)
            E = torch.rad2deg(angle_radians)

            E = E * valid_mask.float()
            
            batch_mean = torch.mean(E).item()
            batch_median = torch.median(E).item()
            batch_rmse = torch.sqrt(torch.mean(E ** 2)).item()
            
            batch_a11 = torch.mean((E < 11.25).float()).item() * 100
            batch_a22 = torch.mean((E < 22.5).float()).item() * 100
            batch_a30 = torch.mean((E < 30.0).float()).item() * 100
            
            self.sum_mean += batch_mean * B
            self.sum_median += batch_median * B
            self.sum_rmse += batch_rmse * B
            self.sum_a11 += batch_a11 * B
            self.sum_a22 += batch_a22 * B
            self.sum_a30 += batch_a30 * B
            self.count_batches += B

    def get_score(self, verbose=False):
        if not self.align_360mtl:
            if not self.all_errors:
                return {'mean': 0.0, 'median': 0.0, 'rmse': 0.0, '11.25': 0.0, '22.5': 0.0, '30': 0.0}

            full_errors = torch.cat(self.all_errors, dim=0)
            mean = torch.mean(full_errors).item()
            median = torch.median(full_errors).item()
            rmse = torch.sqrt(torch.mean(full_errors ** 2)).item()
            acc_11 = (full_errors < 11.25).float().mean().item() * 100
            acc_22 = (full_errors < 22.5).float().mean().item() * 100
            acc_30 = (full_errors < 30.0).float().mean().item() * 100

        else:
            if self.count_batches == 0:
                return {'mean': 0.0, 'median': 0.0, 'rmse': 0.0, '11.25': 0.0, '22.5': 0.0, '30': 0.0}
            
            mean = self.sum_mean / self.count_batches
            median = self.sum_median / self.count_batches
            rmse = self.sum_rmse / self.count_batches
            acc_11 = self.sum_a11 / self.count_batches
            acc_22 = self.sum_a22 / self.count_batches
            acc_30 = self.sum_a30 / self.count_batches

        eval_result = {
            'mean': mean,
            'median': median,
            'rmse': rmse,
            '11.25': acc_11,
            '22.5': acc_22,
            '30': acc_30
        }
        
        if verbose:
            mode = "[360MTL Mode]" if self.align_360mtl else "[Strict Mode]"
            print(f'Results for normals estimation {mode}')
            print(f"Mean: {mean:.4f}, Median: {median:.4f}, RMSE: {rmse:.4f}")
            print(f"<11.25: {acc_11:.2f}%, <22.5: {acc_22:.2f}%, <30: {acc_30:.2f}%")

        return eval_result