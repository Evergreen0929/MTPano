# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image
import pdb

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves',
                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

STANFORD_2D_3D_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
                            "table", "chair", "sofa", "bookcase", "board", "clutter"]

SYNPASS_CATEGORY_NAMES = [
    'building',      # 0
    'fence',         # 1
    'other',         # 2
    'pedestrian',    # 3
    'pole',          # 4
    'road line',     # 5
    'road',          # 6
    'sidewalk',      # 7
    'vegetation',    # 8
    'vehicles',      # 9
    'wall',          # 10
    'traffic sign',  # 11
    'sky',           # 12
    'ground',        # 13
    'bridge',        # 14
    'rail track',    # 15
    'guard rail',    # 16 (表格写为 GroundRail)
    'traffic light', # 17
    'static',        # 18
    'dynamic',       # 19
    'water',         # 20
    'terrain'        # 21
]


ADE20K_CATEGORY_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool", "pillow", "screen", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen", "computer", "swivel", "boat",
    "bar", "arcade", "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van",
    "ship", "fountain", "conveyer", "canopy", "washer", "plaything",
    "swimming", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic", "tray", "ashcan", "fan", "pier", "screen",
    "plate", "monitor", "bulletin", "shower", "radiator", "glass", "clock",
    "flag"
]

class SemsegMeter(object):
    def __init__(self, database, ignore_idx=255):
        ''' "marco" way in ATRC evaluation code.
        '''
        self.empty_classes = []
        if database == 'PASCALContext':
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
             
        elif database == 'NYUD':
            n_classes = 40
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False

        elif database in ['Structured3D', 'Matterport3D']:
            n_classes = 40
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False

            if database == 'Structured3D':
                self.empty_classes = [
                    9,  # bookshelf
                    12, # blinds
                    19, # floor mat
                    20, # clothes
                    22, # books
                    25, # paper
                    26, # towel
                    27, # shower curtain
                    29, # whiteboard
                    30, # person
                    36  # bag
                ]
        
        elif database == 'Stanford2D3D':
            n_classes = 13
            cat_names = STANFORD_2D_3D_NAMES
            has_bg = False
        
        elif database == 'SynPASS':
            n_classes = 22
            cat_names = SYNPASS_CATEGORY_NAMES
            has_bg = False
        
        elif database == 'PanoMTDU':
            n_classes = 150
            cat_names = ADE20K_CATEGORY_NAMES
            has_bg = False

        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

        self.ignore_idx = ignore_idx
        self.database = database

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != self.ignore_idx)
    
        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes
            
    def get_score_v0(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        # eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)


        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = jac #eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result
    
    def get_score_structured3d(self, verbose=True):
        jac = [0] * self.n_classes
        valid_jac_scores = []

        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)
            if i_part not in self.empty_classes:
                valid_jac_scores.append(jac[i_part])

        eval_result = dict()
        eval_result['mIoU'] = np.mean(valid_jac_scores)


        if verbose:
            print('\nSemantic Segmentation mIoU (mean over {0} valid classes): {1:.4f}\n'.format(len(valid_jac_scores), 100 * eval_result['mIoU']))
            class_IoU = jac
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                if i in self.empty_classes:
                    print('{0:s}{1:s}{2:.4f} (SKIPPED in mIoU: Empty class)'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))
                else:
                    print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result
    
    def get_score(self, verbose=True):
        if self.database == 'Structured3D':
            return self.get_score_structured3d(verbose)
        else:
            return self.get_score_v0(verbose)
