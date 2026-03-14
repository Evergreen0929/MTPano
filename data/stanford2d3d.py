# stanford2d3d_mt_dataloader.py (Loads preprocessed data based on corrected JSON)
import os
import sys
import json
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

DB_INFO_DIR = '/sensei-fs/users/yizhouw/projects/collaboration/EEMTL/PanoMTL/data/db_info/'

class Stanford2D3D_MT(data.Dataset):
    """
    Stanford 2D-3D-S dataset for multi-task learning (Panorama only).
    Loads PREPROCESSED panorama images and labels based on JSON splits.
    Handles 'train' and 'test' splits using stanford_train.json/stanford_test.json
    expected in the root directory.

    Assumes data has been preprocessed:
    - RGB:       RGB PNG [0, 255]
    - Semantic:  P-mode PNG (0-13 indices)
    - Depth:     16/32-bit PNG (meters, invalid=0)
    - Normals:   RGB PNG [0, 255] (y-flipped, invalid=(128,128,128))
                   -> Normalized to [-1, 1] here, invalid -> (0,0,0)
    """

    def __init__(self, root, split='train', transform_pano=None, retname=True):
        """
        Args:
            root (str): The root directory containing area_X folders and
                        stanford_train.json/stanford_test.json.
            split (str or list): Which data split to use ('train', 'test').
            transform_pano (callable, optional): Optional transform for panorama data dict.
            retname (bool): If True, returns metadata about the sample.
        """
        self.root = root
        self.transform_pano = transform_pano
        self.retname = retname
        self.split_name = split if isinstance(split, str) else '/'.join(sorted(split))
        db_info_dir = DB_INFO_DIR

        # --- Load Data List from Split JSON ---
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.data_list = []
        print(f"Initializing dataloader for Stanford 2D-3D-S {self.split_name} set(s)...")
        for splt in self.split:
            json_file = os.path.join(db_info_dir, f'stanford_{splt}.json') # Correct filename
            try:
                with open(json_file, 'r') as f:
                    self.data_list.extend(json.load(f))
            except FileNotFoundError:
                raise RuntimeError(f"JSON file for split '{splt}' not found at {json_file}. Run scan_stanford.py after preprocessing.")
            except json.JSONDecodeError:
                raise RuntimeError(f"Error decoding JSON file '{json_file}'.")

        if not self.data_list:
            raise RuntimeError(f"No data found for split(s): {self.split}")

        print(f"Initialized Stanford2D3D_MT Dataset. Found {len(self.data_list)} items for split(s) {self.split_name}.")

    def __getitem__(self, index):
        item_info = self.data_list[index]
        pano_paths = item_info.get('panorama', {}) # Use .get for safety

        # --- Load Preprocessed Panorama Data ---
        pano_sample = self._load_processed_sample(
            pano_paths,
            item_info.get('scene', 'unknown_scene'),
            item_info.get('scan_id', f'unknown_scan_{index}')
        )

        # --- Apply Transforms ---
        if self.transform_pano is not None:
             pano_sample = self.transform_pano(pano_sample)

        # --- Assemble Final Sample Dictionary ---
        sample = {'pano': pano_sample}

        return sample

    def _load_processed_sample(self, paths, scene_id, item_id):
        """Helper function to load a full set of PREPROCESSED data."""
        sample_dict = {}

        # Construct full paths relative to the dataset root
        # Check if key exists in paths dict before joining
        full_img_path = os.path.join(self.root, paths['img']) if 'img' in paths else None
        full_sem_path = os.path.join(self.root, paths['semseg']) if 'semseg' in paths else None
        full_dep_path = os.path.join(self.root, paths['depth']) if 'depth' in paths else None
        full_norm_path = os.path.join(self.root, paths['normal']) if 'normal' in paths else None

        # Load data using simplified loading functions
        _img = self._load_img(full_img_path)
        _semseg = self._load_semseg(full_sem_path)
        _depth = self._load_depth(full_dep_path)
        _normals = self._load_normals(full_norm_path)

        # Check if essential data (image) was loaded
        if _img is None:
             raise FileNotFoundError(f"Failed to load image for {item_id} at {full_img_path}")

        sample_dict['image'] = _img
        sample_dict['semseg'] = _semseg
        sample_dict['depth'] = _depth
        sample_dict['normals'] = _normals

        if self.retname:
            h, w = (_img.shape[0], _img.shape[1])
            sample_dict['meta'] = {
                'scene_id': scene_id,
                'item_id': item_id,
                'img_size': (h, w),
                'processed_paths': paths # Store processed relative paths
            }

        return sample_dict

    def __len__(self):
        return len(self.data_list)

    def _load_img(self, path):
        """Loads an RGB image."""
        _img = Image.open(path).convert('RGB')
        return np.array(_img, dtype=np.float32)

    def _load_semseg(self, path):
        """Loads a semantic segmentation map directly from a paletted PNG."""
        _semseg = Image.open(path)
        _semseg = np.array(_semseg, dtype=np.uint8)
        _semseg = (np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1).astype(np.uint8)
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_depth(self, path):
        """Loads a 16-bit depth map, converts to meters, and sets invalid pixels."""
        _depth = Image.open(path)
        _depth_mm = np.array(_depth, dtype=np.uint16)
        _depth_m = _depth_mm.astype(np.float32) / 512.0
        return np.expand_dims(_depth_m, axis=2)

    def _load_normals(self, path):
        """Loads surface normals, normalizes to [-1, 1], and sets invalid pixels."""
        _normals_img = Image.open(path).convert('RGB')
        _normals_arr = np.array(_normals_img, dtype=np.float32)
        mask_invalid = (_normals_arr == 128).all(axis=2)
        _normals = (_normals_arr / 255.0) * 2.0 - 1.0
        _normals[mask_invalid] = 0
        return _normals

    def __str__(self):
        return f"Stanford 2D-3D-S Multitask Dataset (PREPROCESSED, split={self.split_name}, num_samples={self.__len__()})"


if __name__ == '__main__':
    # --- Example Usage and Verification ---
    # IMPORTANT: Update this path to your Stanford 2D-3D-S base directory
    DATASET_ROOT = '/mnt/localssd/Stanford-2D-3D' # Assumes script is run from Standford-2D-3D base dir

    print("--- Running Dataloader Verification for Stanford2D3D_MT (PREPROCESSED) ---")

    print("\n--- Initializing TEST split ---")
    try:
        test_dataset = Stanford2D3D_MT(root=DATASET_ROOT, split='val')
        print(test_dataset)
        if len(test_dataset) > 0:
            print(f"\nFetching a sample from the TEST dataset (index 0)...")
            sample_outer = test_dataset[0]
            print("Sample keys:", sample_outer.keys())
            assert 'pano' in sample_outer
            print("\n--- Verifying PROCESSED PANO Data ---")
            pano_data = sample_outer['pano']
            pano_img = pano_data.get('image')
            pano_sem = pano_data.get('semseg')
            pano_dep = pano_data.get('depth')
            pano_norm = pano_data.get('normals')
            print("\nShapes, Types, and Value Ranges:")
            if pano_img is not None: print(f"Image:     shape={pano_img.shape}, dtype={pano_img.dtype}")
            if pano_sem is not None: print(f"SemSeg:    shape={pano_sem.shape}, dtype={pano_sem.dtype}, unique={np.unique(pano_sem).tolist()}")
            if pano_dep is not None: print(f"Depth:     shape={pano_dep.shape}, dtype={pano_dep.dtype}, min={pano_dep.min():.3f}, max={pano_dep.max():.3f} (meters)")
            if pano_norm is not None: print(f"Normals:   shape={pano_norm.shape}, dtype={pano_norm.dtype}, min≈{pano_norm[pano_norm != 0].min():.3f}, max≈{pano_norm.max():.3f}")
            if 'meta' in pano_data: print(f"Metadata:  {pano_data['meta']}")
            print("\nBasic verification complete for test sample.")
        else:
            print("Test dataset is empty.")

    except Exception as e:
        print(f"\nAn error occurred during dataloader testing: {e}")
        import traceback
        traceback.print_exc()