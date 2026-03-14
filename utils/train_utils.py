# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


import os, json
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import to_cuda
import torch
from tqdm import tqdm
from utils.test_utils import test_phase, test_phase_pano_pseudo
from utils.panorama_utils import get_camera_matrices, pano_to_perspective_correct, perspective_to_pano_correct
# from utils.panorama_feature_utils import pano_to_perspective_feature, perspective_to_pano_feature
import pdb
import torch.distributed as dist

import random
import math

import torchvision
from utils.panorama_utils import save_depth_map, save_normal_map
from utils.vis_utils import save_semantic_map_for_vis, tensor_to_pil

from utils.aux_label_generator import AuxLabelGenerator

# import torch.profiler

def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def reformat_batch_panomtdu(batch, db_name, iter_count, is_warmup=False):
    if db_name != 'PanoMTDU':
        return batch

    new_pano_dict = batch['pano'].copy()
    source_key_sem_norm = 'merged'
    
    for key in ['semseg', 'normals']:
        if key in batch[source_key_sem_norm]:
            new_pano_dict[key] = batch[source_key_sem_norm][key]

    use_merged_depth = is_warmup or (iter_count < 1000)
    source_key_depth = 'merged' if use_merged_depth else 'random'

    if 'depth' in batch[source_key_depth]:
        new_pano_dict['depth'] = batch[source_key_depth]['depth']


    new_batch = {
        'pano': new_pano_dict
    }
    
    if 'meta' in batch:
        new_batch['meta'] = batch['meta']
        
    return new_batch

def warmup_new_layers(model, train_loader, device, criterion, p, steps=500, aux_gen=None):
    try:
        local_rank = dist.get_rank()
    except:
        local_rank = 0
    is_master = (local_rank == 0)

    if is_master:
        print(f"🔥 [Warmup] Starting Aux Layers Warmup for {steps} steps...")
        print("   -> Training Targets: aggregator_inv/var, aux_heads, ext_heads")
        print("   -> Frozen: Backbone, GroupRefiner, Bridge, Injectors, Main Heads")

    real_model = model.module if hasattr(model, 'module') else model
    
    # 1. 保存原始 requires_grad 状态并全部冻结
    grad_state_dict = {}
    for name, param in real_model.named_parameters():
        grad_state_dict[name] = param.requires_grad
        param.requires_grad = False 
    
    # 2. 定义需要 Warmup 的“新层” (修正为 BFE 类的属性)
    # 根据你的要求：只给 ScaleAggregator 和 AuxTaskHead
    trainable_modules = [
        real_model.aggregator_inv,
        real_model.aggregator_var,
        real_model.aux_heads,
        real_model.extended_heads
    ]
    
    # 解冻目标模块
    active_params = []
    for mod in trainable_modules:
        for param in mod.parameters():
            param.requires_grad = True
            active_params.append(param)
            
    # 3. 创建临时优化器 (LR 可以稍微大一点，比如 1e-3 or 5e-4，让它快速找到感觉)
    optimizer = torch.optim.Adam(active_params, lr=1e-3)
    
    model.train()
    
    if dist.is_initialized():
        dist.barrier()
        
    iter_count = 0
    data_iter = iter(train_loader)
    
    if is_master:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=steps, desc="Warmup Aux", leave=False)
        except ImportError:
            pbar = None

    while iter_count < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        batch = reformat_batch_panomtdu(batch, p['train_db_name'], iter_count=0, is_warmup=True)
        batch = to_cuda(batch)

        if isinstance(batch, dict) and 'pano' in batch:
            images = batch['pano']['image'].to(device)
            targets = batch['pano']
        else:
            images = batch['image'].to(device)
            targets = batch
        
        with torch.no_grad():
            gt_gradients = aux_gen.generate_gradient_map(images)
            gt_sdf = aux_gen.generate_sdf_map(images)
            ext_gt_dict = {'grad': gt_gradients, 'sdf': gt_sdf}
            if 'depth' in p.TASKS.NAMES:
                gt_point_map = aux_gen.generate_point_map(targets['depth'])
                ext_gt_dict['point'] = gt_point_map

        # Forward
        output = model(images)
        
        # 计算 Loss
        loss_dict = criterion(output, targets, ext_gt=ext_gt_dict, tasks=p.TASKS.NAMES, clamp_depth=False)
        loss_total = loss_dict['total']
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        iter_count += 1
        
        if is_master and pbar:
            log_dict = {'total': f"{loss_total.item():.2f}"}
            for k, v in loss_dict.items():
                if k.startswith('aux_'):
                    short_key = k.replace('aux_', 'a_')[:5] 
                    log_dict[short_key] = f"{v.item():.2f}"
                if k.startswith('ext_'):
                    short_key = k.replace('ext_', 'e_')[:5] 
                    log_dict[short_key] = f"{v.item():.2f}"
            pbar.set_postfix(log_dict)
            pbar.update(1)
            
    if is_master and pbar:
        pbar.close()
        
    if dist.is_initialized():
        dist.barrier()
        
    # 6. 恢复原始状态
    for name, param in real_model.named_parameters():
        if name in grad_state_dict:
            param.requires_grad = grad_state_dict[name]

    if is_master:
        print(f"✅ Aux Layers Warmup Done. Loss: {loss_total.item():.2f}")
        for k, v in loss_dict.items():
            print(f"{k}: {v.item():.2f}") 


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, tb_writer_test_pano, iter_count, model_t=None, criterion_aug=None, aux_gen=None):
    if iter_count == 0 and p['train_mode'] == 'train_gt' and p['model'] in ['TransformerBFE-DINO-DPT', 'TransformerBFE-DINO-DPT-Abl']:
        device = torch.device("cuda", args.local_rank)
        if p['train_db_name'] == 'PanoMTDU':
            warmup_steps = 1000  
        elif p['train_db_name'] == 'Stanford2D3D':
            warmup_steps = 200
        elif p['train_db_name'] == 'Matterport3D':
            warmup_steps = 100
        else:
            warmup_steps = 500
        warmup_new_layers(model, train_loader, device, criterion, p, steps=warmup_steps, aux_gen=aux_gen)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    """ Vanilla training with fixed loss weights """
    model.train() 

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        cpu_batch = reformat_batch_panomtdu(cpu_batch, p['train_db_name'], iter_count, is_warmup=False)
        batch = to_cuda(cpu_batch)

        if p['train_db_name'] in ['Structured3D', 'Stanford2D3D', 'Matterport3D', 'SynPASS', 'Deep360', 'PanoMTDU']:
            pano_rgb = batch['pano']['image'] 
            pano_semantic = batch['pano']['semseg'] if 'semseg' in batch['pano'].keys() else None
            pano_depth = batch['pano']['depth'] if 'depth' in batch['pano'].keys() else None
            pano_normal = batch['pano']['normals'] if 'normals' in batch['pano'].keys() else None

            if p['train_mode'] == 'train_gt':
                images = pano_rgb
                with torch.no_grad():
                    gt_gradients = aux_gen.generate_gradient_map(images)
                    gt_sdf = aux_gen.generate_sdf_map(images)
                    ext_gt_dict = {'grad': gt_gradients, 'sdf': gt_sdf}
                    if 'depth' in p.TASKS.NAMES:
                        gt_point_map = aux_gen.generate_point_map(pano_depth)
                        ext_gt_dict['point'] = gt_point_map
                output = model(images)
                iter_count += 1
                loss_dict = criterion(output, batch['pano'], ext_gt=ext_gt_dict, tasks=p.TASKS.NAMES)
            
            else:
                raise NotImplementedError("Only 'train_gt' mode is implemented for panorama datasets.")

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()
        scheduler.step()
        
        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))

            if p['train_db_name'] in ['Structured3D', 'Stanford2D3D', 'Matterport3D', 'SynPASS', 'Deep360', 'PanoMTDU']:
                curr_result_pano = test_phase_pano_pseudo(p, test_dataloader, model, criterion, iter_count)
                tb_update_perf(p, tb_writer_test_pano, curr_result_pano, iter_count)
                print('Evaluate panorama results at iteration {}: \n'.format(iter_count))
                print(curr_result_pano)

                with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                    json.dump(curr_result_pano, f, indent=4)
                    

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            
            # print(f'Full Model (Inference) saved to: {full_model_path}')
            print('Checkpoint finishs.')
            model.train() # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)

def tb_update_feat_perf(p, tb_writer_test, curr_result, cur_iter):
    sum_loss = 0.0
    for k, v in curr_result.items():
        tb_writer_test.add_scalar(f'feat_perf/{k}', v, cur_iter)
        sum_loss += v
    tb_writer_test.add_scalar('feat_perf/avg_loss', sum_loss / len(curr_result), cur_iter)
