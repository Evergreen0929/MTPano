import os
import json
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

DB_INFO_DIR = '/mnt/localssd/code/PanoMTL/data/db_info/'

class Structured3D_MT(data.Dataset):
    """
    Structured3D dataset for multi-task learning, inspired by the NYUD_MT structure.
    Loads panorama and a randomly selected perspective view for each sample.
    Handles 'train' and 'val' splits based on pre-generated JSON files.
    
    Tasks included:
    - Semantic Segmentation
    - Depth Prediction
    - Surface Normals
    """

    def __init__(self, root, split='train', transform=None, transform_pano=None, retname=True):
        """
        Args:
            root (str): The root directory of the Structured3D dataset.
            db_info_dir (str): Path to the directory containing the split JSON files.
            split (str or list): Which data split to use ('train', 'val').
            transform (callable, optional): Optional transform to be applied on a sample.
            retname (bool): If True, returns metadata about the sample.
        """
        self.root = root
        self.transform = transform
        self.transform_pano = transform_pano
        self.retname = retname
        db_info_dir = DB_INFO_DIR
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # Load the list of valid data items from the corresponding JSON file(s)
        self.data_list = []
        print(f"Initializing dataloader for Structured3D {'/'.join(self.split)} set(s)...")
        for splt in self.split:
            json_file = os.path.join(db_info_dir, f'structured3d_pairs_{splt}.json')
            try:
                with open(json_file, 'r') as f:
                    self.data_list.extend(json.load(f))
            except FileNotFoundError:
                raise RuntimeError(f"JSON file for split '{splt}' not found at {json_file}")
        
        if not self.data_list:
            raise RuntimeError(f"No data found for split(s): {self.split}")

        print(f"Initialized Structured3D_MT Dataset. Found {len(self.data_list)} items for split(s) {'/'.join(self.split)}.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary containing the panoramic and perspective data.
        """
        # Get the metadata for the current item
        item_info = self.data_list[index]
        
        # --- 1. Load Panorama Data ---
        pano_paths = item_info['panorama']
        pano_sample = self._load_sample(pano_paths, item_info['scene'], item_info['per_scene_id'])

        # # --- 2. Randomly Select and Load Perspective Data ---
        # persp_keys = [k for k in item_info if k.startswith('perspective_')]
        # if not persp_keys:
        #     persp_sample = {'image': None, 'semseg': None, 'depth': None, 'normals': None, 'meta': {}}
        # else:
        #     if self.split == ['train']:
        #         random_persp_key = random.choice(persp_keys)
        #     elif self.split == ['val']:
        #         random_persp_key = persp_keys[0]
        #     persp_paths = item_info[random_persp_key]
        #     persp_id = f"{item_info['per_scene_id']}_{random_persp_key.split('_')[-1]}"
        #     persp_sample = self._load_sample(persp_paths, item_info['scene'], persp_id)

        # --- 3. Assemble Final Sample Dictionary ---
        # if self.transform is not None:
        #     sample = self.transform(persp_sample)
        if self.transform_pano is not None:
            sample = self.transform_pano(pano_sample)
        sample = {'pano': pano_sample}

        return sample

    def _load_sample(self, paths, scene_id, item_id):
        """A helper function to load a full set of image and labels."""
        sample_dict = {}
        
        full_img_path = os.path.join(self.root, paths['img'])
        full_sem_path = os.path.join(self.root, paths['semseg'])
        full_dep_path = os.path.join(self.root, paths['depth'])
        full_norm_path = os.path.join(self.root, paths['normal'])
        
        _img = self._load_img(full_img_path)
        _semseg = self._load_semseg(full_sem_path)
        _depth = self._load_depth(full_dep_path)
        _normals = self._load_normals(full_norm_path)
        
        sample_dict['image'] = _img
        sample_dict['semseg'] = _semseg
        sample_dict['depth'] = _depth
        sample_dict['normals'] = _normals

        if self.retname:
            sample_dict['meta'] = {
                'scene_id': scene_id,
                'item_id': item_id,
                'img_size': (_img.shape[0], _img.shape[1])
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
        _depth_m = _depth_mm.astype(np.float32) / 1000.0
        return np.expand_dims(_depth_m, axis=2)

    def _load_normals(self, path):
        """Loads surface normals, normalizes to [-1, 1], and sets invalid pixels."""
        _normals_img = Image.open(path).convert('RGB')
        _normals_arr = np.array(_normals_img, dtype=np.float32)
        _normals = (_normals_arr / 255.0) * 2.0 - 1.0
        return _normals

    def __str__(self):
        return f"Structured3D Multitask Dataset (split={'/'.join(self.split)}, num_samples={self.__len__()})"


if __name__ == '__main__':
    # --- Example Usage and Verification ---
    
    # IMPORTANT: Update these paths to match your system
    DATASET_ROOT = '/mnt/localssd/Structured3D/Structured3D'
    
    print("--- Running Dataloader Verification for Structured3D_MT ---")
    
    if not os.path.exists(DATASET_ROOT) or not os.path.exists(DB_INFO_DIR):
        print("\nERROR: Please update DATASET_ROOT and DB_INFO_DIR paths in the script to run the test.")
    else:
        # 1. Initialize the validation dataset
        print("\n--- Initializing VAL split ---")
        val_dataset = Structured3D_MT(root=DATASET_ROOT, split='val')
        print(val_dataset)

        # 2. Initialize the training dataset to confirm it loads
        print("\n--- Initializing TRAIN split ---")
        train_dataset = Structured3D_MT(root=DATASET_ROOT, split='train')
        print(train_dataset)
        
        # 3. Get a single sample from the validation set for detailed checking
        print("\nFetching a sample from the VAL dataset (index 0)...")
        sample = val_dataset[0]
        
        # 4. Verify the top-level structure
        print("Sample keys:", sample.keys())
        assert 'pano' in sample and 'persp' in sample
        
        # --- 5. Verify Panorama Data Content ---
        print("\n--- Verifying PANO Data Shapes and Types ---")
        pano_data = sample['pano']
        pano_img = pano_data['image']
        pano_sem = pano_data['semseg']
        pano_dep = pano_data['depth']
        pano_norm = pano_data['normals']
        
        valid_pano_dep = pano_dep[pano_dep != 255]
        pano_dep_min = valid_pano_dep.min() if valid_pano_dep.size > 0 else 'N/A'
        pano_dep_max = valid_pano_dep.max() if valid_pano_dep.size > 0 else 'N/A'
        
        print(f"Image:         shape={pano_img.shape}, dtype={pano_img.dtype}")
        print(f"SemSeg:        shape={pano_sem.shape}, dtype={pano_sem.dtype}, unique_vals={np.unique(pano_sem).tolist()}")
        print(f"Depth:         shape={pano_dep.shape}, dtype={pano_dep.dtype}, min={pano_dep_min:.3f}, max={pano_dep_max:.3f}")
        print(f"Normals:       shape={pano_norm.shape}, dtype={pano_norm.dtype}, min={pano_norm[pano_norm != 255].min():.3f}, max={pano_norm[pano_norm != 255].max():.3f}")
        print(f"Metadata:      {pano_data['meta']}")
        
        # --- 6. Verify Perspective Data Content ---
        print("\n--- Verifying PERSP Data Shapes and Types ---")
        persp_data = sample['persp']
        persp_img = persp_data['image']
        persp_sem = persp_data['semseg']
        persp_dep = persp_data['depth']
        persp_norm = persp_data['normals']

        valid_persp_dep = persp_dep[persp_dep != 255]
        persp_dep_min = valid_persp_dep.min() if valid_persp_dep.size > 0 else 'N/A'
        persp_dep_max = valid_persp_dep.max() if valid_persp_dep.size > 0 else 'N/A'

        print(f"Image:         shape={persp_img.shape}, dtype={persp_img.dtype}")
        print(f"SemSeg:        shape={persp_sem.shape}, dtype={persp_sem.dtype}, unique_vals={np.unique(persp_sem).tolist()}")
        print(f"Depth:         shape={persp_dep.shape}, dtype={persp_dep.dtype}, min={persp_dep_min:.3f}, max={persp_dep_max:.3f}")
        print(f"Normals:       shape={persp_norm.shape}, dtype={persp_norm.dtype}, min={persp_norm[persp_norm != 255].min():.3f}, max={persp_norm[persp_norm != 255].max():.3f}")
        print(f"Metadata:      {persp_data['meta']}")
        
        # --- 7. Specific Check for Ignore Index ---
        print("\n--- Verifying Ignore Index Application (on Pano Depth) ---")
        original_depth_path = os.path.join(DATASET_ROOT, val_dataset.data_list[0]['panorama']['depth'])
        original_depth_mm = np.array(Image.open(original_depth_path))
        num_zeros_original = np.sum(original_depth_mm == 0)
        num_255_processed = np.sum(pano_dep == 255)
        print(f"Original panorama depth had {num_zeros_original} zero pixels.")
        print(f"Processed panorama depth has {num_255_processed} pixels with ignore_index=255.")
        assert num_zeros_original == num_255_processed
        print("Depth ignore index verified successfully!")

        print("\nVerification complete. The Structured3D_MT dataloader is ready for train/val splits.")