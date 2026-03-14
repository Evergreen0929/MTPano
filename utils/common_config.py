# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    DPTConfig,
    DPTForDepthEstimation,
    DINOv3ViTConfig,
)


def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'dinov3L':
        from transformers import DINOv3ViTModel
        
        backbone = DINOv3ViTModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m", 
            use_safetensors=True
        )
        
        backbone_channels = [1024 for _ in range(4)]
        p.backbone_channels = backbone_channels
        
        # ViT-L/16 的 Patch size 是 16
        patch_size = 16 # backbone.config.patch_size
        p.spatial_dim = [[p.TRAIN.SCALE[0] // patch_size, p.TRAIN.SCALE[1] // patch_size] for _ in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT

    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """

    if p['head'] == 'mlp':
        from models.transformers.transformer_decoder import MLPHead
        return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == 'conv':
        from models.transformers.taskprompter import ConvHead
        return ConvHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == 'dpt_head':
        from transformers import DPTForDepthEstimation, DPTForSemanticSegmentation
        print(f"Loading pretrained DPT head for task: {task}")
        # from transformers import DPTConfig
        from transformers.models.dpt.modeling_dpt import DPTDepthEstimationHead, DPTSemanticSegmentationHead
        
        if task == 'depth':
            dpt_model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-large", 
                use_safetensors=True
            )
            return dpt_model.head
        
        elif task == 'semseg':
            num_labels = p.TASKS.NUM_OUTPUT[task]
            print(f"Configuring semseg head for {num_labels} classes.")
            
            config = DPTConfig.from_pretrained("Intel/dpt-large-ade")
            config.num_labels = num_labels
            print('Use feature from neck:', config.head_in_index)
            
            return DPTSemanticSegmentationHead(config)
            
        elif task == 'normals':
            from models.transformer_net import DPTNormalsHead
            print("Initializing a new DPTNormalsHead (from scratch) using DPTDepthEstimationHead's architecture.")
            
            config = DPTConfig.from_pretrained("Intel/dpt-large")
            print('Use feature from neck:', config.head_in_index)

            return DPTNormalsHead(config)


def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)
    if p['model'] == 'TransformerBFE-DINO-DPT':
        from models.transformer_net import TransformerBaselineBFE
        feat_channels = p.backbone_channels[-1] 
        heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
        model = TransformerBaselineBFE(p, backbone, backbone_channels, heads)
    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] in ['Structured3D', 'Stanford2D3D', 'Matterport3D', 'SynPASS', 'Deep360', 'PanoMTDU']:
        train_transforms_pano = torchvision.transforms.Compose([ # from ATRC
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AddIgnoreRegions(),
                transforms.ToTensor(),
            ])

        valid_transforms_pano = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.PANO_SCALE),
            # transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        return train_transforms_pano, valid_transforms_pano

    else:
        return None, None


def get_train_dataset(p, transforms=None, transforms_pano=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))
    
    if db_name == 'Structured3D':
        from data.structured3d import Structured3D_MT
        database = Structured3D_MT(p.db_paths['Structured3D_MT'], split=['train'], transform=transforms, transform_pano=transforms_pano, retname=True)
    
    elif db_name == 'Stanford2D3D':
        from data.stanford2d3d import Stanford2D3D_MT
        database = Stanford2D3D_MT(p.db_paths['Stanford2D3D_MT'], split=['train'], transform_pano=transforms_pano, retname=True)
    
    elif db_name == 'Matterport3D':
        from data.matterport3d import Matterport3D_MT
        database = Matterport3D_MT(p.db_paths['Matterport3D_MT'], split=['train'], transform=transforms_pano, retname=True)
    
    elif db_name == 'SynPASS':
        from data.synpass import SynPASS_Seg
        database = SynPASS_Seg(p.db_paths['SynPASS'], split=['train'], transform=transforms_pano, retname=True)
    
    elif db_name == 'Deep360':
        from data.deep360 import Deep360_Depth
        database = Deep360_Depth(p.db_paths['Deep360'], split=['training'], transform=transforms_pano, retname=True)

    elif db_name == 'PanoMTDU':
        from data.pano_mtdu import PanoMTDU
        database = PanoMTDU(p.db_paths['PanoMTDU'], split=['train'], transform_pano=transforms_pano, retname=True)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None, transforms_pano=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))
    
    if db_name == 'Structured3D':
        from data.structured3d import Structured3D_MT
        database = Structured3D_MT(p.db_paths['Structured3D_MT'], split=['val'], transform=transforms, transform_pano=transforms_pano, retname=True)
    
    elif db_name == 'Stanford2D3D':
        from data.stanford2d3d import Stanford2D3D_MT
        database = Stanford2D3D_MT(p.db_paths['Stanford2D3D_MT'], split=['val'], transform_pano=transforms_pano, retname=True)
    
    elif db_name == 'Matterport3D':
        from data.matterport3d import Matterport3D_MT
        database = Matterport3D_MT(p.db_paths['Matterport3D_MT'], split=['test'], transform=transforms_pano, retname=True)
    
    elif db_name == 'SynPASS':
        from data.synpass import SynPASS_Seg
        database = SynPASS_Seg(p.db_paths['SynPASS'], split=['test'], transform=transforms_pano, retname=True)
    
    elif db_name == 'Deep360':
        from data.deep360 import Deep360_Depth
        database = Deep360_Depth(p.db_paths['Deep360'], split=['testing'], transform=transforms_pano, retname=True)

    elif db_name == 'PanoMTDU':
        from data.pano_mtdu import PanoMTDU
        database = PanoMTDU(p.db_paths['PanoMTDU'], split=['test'], transform_pano=transforms_pano, retname=True)

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss, DiceCELoss
        # criterion = CrossEntropyLoss(ignore_index=p.ignore_index)
        criterion = DiceCELoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss, L1FreqAwareLoss
        criterion = L1FreqAwareLoss(task, normalize=True, ignore_index=p.ignore_index, edge_weight=0.1, freq_weight=1e-4)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss, L1FreqAwareLoss
        criterion = L1FreqAwareLoss(task)

    else:
        criterion = None

    return criterion


def get_criterion(p, use_ssl=False):
    if use_ssl:
        from losses.loss_schemes import SSLLoss
        from losses.loss_functions import L2Loss
        return SSLLoss(p, L2Loss(normalize=True))
    else:
        from losses.loss_schemes import MultiTaskLoss
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])


    elif p['optimizer'] == 'adamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, params), lr=p['optimizer_kwargs']['lr'],
                                      betas=(p['optimizer_kwargs']['betas']['a'], p['optimizer_kwargs']['betas']['b']),
                                      weight_decay=p['optimizer_kwargs']['weight_decay'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler
    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)
    elif p.scheduler == 'step':
        scheduler = torch.optim.MultiStepLR(optimizer, milestones=p.scheduler_kwargs.milestones, gamma=p.scheduler_kwargs.lr_decay_rate)

    return scheduler, optimizer