# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os
from utils.panorama_utils import get_camera_matrices, pano_to_perspective_correct, perspective_to_pano_correct
from utils.panorama_feature_utils import pano_to_perspective_feature
from utils.vis_utils import visualize_results, visualize_feature_alignment
# from evaluation.eval_semseg import TTASemsegWrapper
import numpy as np

def reformat_batch_panomtdu_test(batch, db_name):
    if db_name != 'PanoMTDU':
        return batch
    new_pano_dict = batch['pano'].copy()

    if 'merged' in batch:
        new_pano_dict.update(batch['merged'])

    new_batch = {
        'pano': new_pano_dict
    }

    if 'meta' in batch:
        new_batch['meta'] = batch['meta']

    return new_batch

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    tasks = p.TASKS.NAMES

    performance_meter = PerformanceMeter(p, tasks)

    model.eval()

    if 'edge' in tasks:
        tasks_to_save = ['edge']
        save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
        for save_dir in save_dirs.values():
            mkdir_if_missing(save_dir)
    else:
        tasks_to_save = []
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            output = model.module(images)
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks}, 
                                    {t: targets[t] for t in tasks})

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, batch, output, save_dirs, task, epoch=epoch)


    eval_results = performance_meter.get_score(verbose = True)

    return eval_results


@torch.no_grad()
def test_phase_pano_pseudo(p, test_loader, model, criterion, epoch, log_wandb=True):
    tasks = p.TASKS.NAMES

    performance_meter_pano = PerformanceMeter(p, tasks)

    model.eval()

    if 'edge' in tasks:
        tasks_to_save = ['edge']
        save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
        for save_dir in save_dirs.values():
            mkdir_if_missing(save_dir)
    else:
        tasks_to_save = []

    # PERSP_H, PERSP_W = p.TEST.SCALE
    PERSP_H, PERSP_W = p.TEST.PERSP_SCALE 
    H_FOV = 100.0
    YAW = 180.0
    PITCH_base = 0.0
    PITCH = PITCH_base + 180.0
    
    ph, pw = int(PERSP_H), int(PERSP_W)

    ignore_value = 255
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        batch = reformat_batch_panomtdu_test(batch, p['train_db_name'])
        with torch.no_grad():
            pano_rgb = batch['pano']['image'].cuda(non_blocking=True) 
            # pano_semantic = batch['pano']['semseg'].cuda(non_blocking=True) 
            # pano_depth = batch['pano']['depth'].cuda(non_blocking=True) 
            # pano_normal = batch['pano']['normals'].cuda(non_blocking=True) 

            # 动态获取当前 batch 的 B, H, W 和所在设备
            B, _, H_img, W_img = pano_rgb.shape
            device = pano_rgb.device
            
            # Semantic: (B, 1, H, W), 用 255 占位 (Ignore Index)，防止参与任何 Loss 或 Metric 计算
            pano_semantic = batch['pano']['semseg'].cuda(non_blocking=True) if 'semseg' in tasks else \
                            torch.full((B, 1, H_img, W_img), 255, dtype=torch.float32, device=device)
                            
            # Depth: (B, 1, H, W), 用 0 占位 (代表无效深度)
            pano_depth = batch['pano']['depth'].cuda(non_blocking=True) if 'depth' in tasks else \
                         torch.zeros((B, 1, H_img, W_img), dtype=torch.float32, device=device)
                         
            # Normals: (B, 3, H, W), 用 0 占位 (后续 norm=0 会触发 mask 并被设为 ignore_value)
            pano_normal = batch['pano']['normals'].cuda(non_blocking=True) if 'normals' in tasks else \
                          torch.zeros((B, 3, H_img, W_img), dtype=torch.float32, device=device)

            pano_depth[pano_depth == 255] = 0
            pano_normal[pano_normal == 255] = 0

            persp_rgb, persp_semantic, persp_z_depth, persp_cam_normal = pano_to_perspective_correct(
                pano_rgb, pano_semantic, pano_depth, pano_normal, H_FOV, YAW, PITCH, ph, pw
            )

            targets_persp_synth = {}

            # a) 处理 'depth' (persp_z_depth)
            persp_z_depth_masked = persp_z_depth.clone()
            if 'depth' in tasks:
                depth_zero_mask = (persp_z_depth == 0)
                persp_z_depth_masked[depth_zero_mask] = ignore_value
                targets_persp_synth['depth'] = persp_z_depth_masked
            
            # b) 处理 'normals' (persp_cam_normal)
            persp_cam_normal_masked = persp_cam_normal.clone()
            if 'normals' in tasks:
                norm = torch.linalg.norm(persp_cam_normal, dim=1, keepdim=True)
                persp_cam_normal = persp_cam_normal / (norm + 1e-8)
                normals_zero_norm_mask = (norm == 0)
                normals_ignore_mask_c = normals_zero_norm_mask.expand_as(persp_cam_normal)
                persp_cam_normal_masked[normals_ignore_mask_c] = ignore_value
                targets_persp_synth['normals'] = persp_cam_normal_masked

            # c) 语义
            if 'semseg' in tasks:
                targets_persp_synth['semseg'] = persp_semantic
            
            # --- 2. 模型前向传播 (使用合成透视 RGB) ---
            images_for_model = pano_rgb
            if hasattr(model, 'module'):
                output = model.module(images_for_model)
            else:
                output = model(images_for_model)

            PANO_H, PANO_W = pano_rgb.shape[-2:] # 获取全景图尺寸
            
            # 需要从 output 中提取预测结果，并确保格式正确 (One-Hot for semseg)
            persp_pred_rgb = persp_rgb
            persp_pred_semantic = persp_semantic # (B, C, H, W)
            persp_pred_z_depth = persp_z_depth  # (B, 1, H, W)
            persp_pred_cam_normal = persp_cam_normal # (B, 3, H, W)

            pred_persp_place_holder = {
                'semseg': persp_pred_semantic,
                'depth': persp_pred_z_depth,
                'normals': persp_pred_cam_normal
            }

            # --- 应用忽略区域 (255.0) 到 Warped Pano 预测和原始 Pano 标签 ---
            
            target_pano_masked = {}
            
            if 'depth' in tasks:
                target_depth = pano_depth.clone()
                depth_zero_mask_t = (target_depth == 0)
                target_depth[depth_zero_mask_t] = ignore_value
                target_pano_masked['depth'] = target_depth

            if 'normals' in tasks:
                target_normals = pano_normal.clone()
                norm_t = torch.linalg.norm(target_normals, dim=1, keepdim=True)
                target_normals = target_normals / (norm_t + 1e-8)
                normals_zero_norm_mask_t = (norm_t == 0)
                normals_ignore_mask_c_t = normals_zero_norm_mask_t.expand_as(target_normals)
                target_normals[normals_ignore_mask_c_t] = ignore_value
                target_pano_masked['normals'] = target_normals

            if 'semseg' in tasks:
                target_semseg = pano_semantic.clone()
                target_pano_masked['semseg'] = target_semseg
            
            performance_meter_pano.update(
                {t: get_output(output[t], t) for t in tasks}, 
                {t: target_pano_masked[t] for t in tasks}
            )

            if i < 3 and log_wandb:
                visualize_results(
                    p, 
                    batch, 
                    pred_persp_place_holder, 
                    output, 
                    target_pano_masked,
                    persp_rgb, 
                    targets_persp_synth, 
                    i, 
                    epoch
                )

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, batch, output, save_dirs, task, epoch=epoch)


    eval_results_pano = performance_meter_pano.get_score(verbose = True)

    return eval_results_pano