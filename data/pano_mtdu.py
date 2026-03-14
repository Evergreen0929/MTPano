import os
import json
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

# 根据你的代码片段，这里使用你指定的 DB_INFO_DIR
DB_INFO_DIR = '/mnt/localssd/code/PanoMTL/data/db_info'

class PanoMTDU(data.Dataset):
    """
    PanoMTDU dataset: Loads Pano + Random Pseudo Label + Merged Pseudo Label.
    """

    def __init__(self, root, split='train', transform_pano=None, retname=True):
        """
        Args:
            root (str): Root of the generated dataset (containing img, semseg, depth, normals).
            split (str or list): 'train', 'val', or list of splits.
            transform_pano (callable, optional): Transform to be applied on the WHOLE sample (image + all labels).
            retname (bool): Whether to return metadata.
        """
        self.root = root
        self.transform_pano = transform_pano  # 只保留这一个 transform
        self.retname = retname
        db_info_dir = DB_INFO_DIR
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # 1. Load the Split File
        self.data_list = []
        print(f"Initializing dataloader for PanoMTDU {'/'.join(self.split)} set(s)...")
        for splt in self.split:
            # 修正文件名格式以匹配你的要求
            json_file = os.path.join(db_info_dir, f'panomtdu_{splt}.json')
            try:
                with open(json_file, 'r') as f:
                    self.data_list.extend(json.load(f))
            except FileNotFoundError:
                raise RuntimeError(f"JSON file for split '{splt}' not found at {json_file}")
        
        if not self.data_list:
            raise RuntimeError(f"No data found for split(s): {self.split}")
            
        print(f"Initialized PanoMTDU Dataset ({'/'.join(self.split)}). Loaded {len(self.data_list)} samples.")

    def __getitem__(self, index):
        """
        Returns a nested dict. 
        Crucially, transforms are applied to a flat dict first to ensure consistency.
        """
        # 1. Get ID and Paths
        safe_id = self.data_list[index]
        
        # Paths Setup
        img_path = os.path.join(self.root, 'img', f"{safe_id}.jpg")
        
        # --- Random View Selection ---
        rand_idx = random.randint(0, 31)
        rand_name = f"label_{rand_idx}.png"
        
        # --- Merged View Name ---
        merged_name = "merged.png"
        conf_name = "merged_confidence.png"

        # 2. Load Data (Load everything as numpy arrays first)
        pano_img = self._load_img(img_path)
        
        # Random View Labels
        rand_sem = self._load_semseg(os.path.join(self.root, 'semseg', safe_id, rand_name))
        rand_dep = self._load_depth(os.path.join(self.root, 'depth', safe_id, rand_name))
        rand_norm = self._load_normals(os.path.join(self.root, 'normals', safe_id, rand_name))
        
        # Merged View Labels & Confidence
        merged_sem = self._load_semseg(os.path.join(self.root, 'semseg', safe_id, merged_name))
        merged_dep = self._load_depth(os.path.join(self.root, 'depth', safe_id, merged_name))
        merged_norm = self._load_normals(os.path.join(self.root, 'normals', safe_id, merged_name))
        merged_conf = self._load_conf(os.path.join(self.root, 'semseg', safe_id, conf_name))

        # 3. Construct a FLAT Dictionary for Transform
        # 这一点至关重要：为了让 RandomHorizontalPanoFlip 对所有数据生效，
        # 我们必须把它们放在同一个字典的一级目录下。
        # 使用特定的 key 名称以防覆盖，transform 处理完后再解包。
        full_sample = {
            'image': pano_img,
            
            # Random View Data
            'semseg': rand_sem,
            'depth': rand_dep,
            'normals': rand_norm,
            
            # Merged View Data (使用不同 Key 名)
            'merged_semseg': merged_sem,
            'merged_depth': merged_dep,
            'merged_normals': merged_norm,
            'merged_conf': merged_conf
        }

        # 4. Apply Transform
        # 这里的 self.transform_pano 应该是 Compose([...])
        # 其中包含的 RandomHorizontalPanoFlip 会遍历 items() 并翻转所有 array
        if self.transform_pano is not None:
            full_sample = self.transform_pano(full_sample)

        # 5. Restructure (Unpack) into Final Nested Dictionary
        # 注意：如果 transform 把 numpy 转成了 tensor，这里取出来的就是 tensor
        sample = {
            'pano': {'image': full_sample['image']},
            'random': {
                'semseg': full_sample['semseg'],
                'depth': full_sample['depth'],
                'normals': full_sample['normals']
            },
            'merged': {
                'semseg': full_sample['merged_semseg'],
                'depth': full_sample['merged_depth'],
                'normals': full_sample['merged_normals'],
                'conf': full_sample['merged_conf']
            }
        }

        if self.retname:
            sample['meta'] = {
                'scene_id': safe_id,
                'random_view_idx': rand_idx,
                # 注意处理 transform 可能改变尺寸的情况 (虽然 Pano 通常只是 Flip)
                'img_size': (full_sample['image'].shape[-2], full_sample['image'].shape[-1]) 
                            if hasattr(full_sample['image'], 'shape') else (512, 1024)
            }

        return sample

    def __len__(self):
        return len(self.data_list)

    def _load_img(self, path):
        """Loads RGB image."""
        _img = Image.open(path).convert('RGB')
        return np.array(_img, dtype=np.float32)

    def _load_semseg(self, path):
        """
        Loads semseg. 
        Gen script saves 255 as ignore, classes 0-149.
        """ 
        if not os.path.exists(path):
            # Fallback safety
            return np.full((512, 1024, 1), 255, dtype=np.uint8)
        _semseg = Image.open(path)
        _semseg = np.array(_semseg, dtype=np.uint8)
        return np.expand_dims(_semseg, axis=2)

    def _load_depth(self, path):
        """
        Loads depth (uint16 mm -> float m).
        0 is invalid/ignore.
        """
        if not os.path.exists(path):
            return np.zeros((512, 1024, 1), dtype=np.float32)

        _depth = Image.open(path)
        _depth_mm = np.array(_depth, dtype=np.uint16)
        _depth_m = _depth_mm.astype(np.float32) / 1000.0
        return np.expand_dims(_depth_m, axis=2)

    def _load_normals(self, path):
        """
        Loads normals. 
        Gen script: invalid is [128,128,128].
        Decoding: (x/255)*2 - 1.
        """
        if not os.path.exists(path):
            return np.zeros((512, 1024, 3), dtype=np.float32)

        _normals_img = Image.open(path).convert('RGB')
        _normals_arr = np.array(_normals_img, dtype=np.float32)
        mask_invalid = (_normals_arr == 128).all(axis=2)
        _normals = (_normals_arr / 255.0) * 2.0 - 1.0
        _normals[mask_invalid] = 0
        return _normals

    def _load_conf(self, path):
        """
        Loads confidence map.
        Assumes saved as grayscale image (0-255).
        Returns float [0, 1].
        """
        if not os.path.exists(path):
            return np.ones((512, 1024, 1), dtype=np.float32)
            
        _conf = Image.open(path)
        _conf_arr = np.array(_conf, dtype=np.float32)
        _conf_norm = _conf_arr / 255.0
        return np.expand_dims(_conf_norm, axis=2)

if __name__ == '__main__':
    # --- Example Usage and Verification ---
    
    # IMPORTANT: Update these paths to match your system
    DATASET_ROOT = '/mnt/localssd/PanoPseudoLabels'
    # 确保 split json 文件在这个目录下
    DB_INFO_DIR = '/mnt/localssd/code/PanoMTL/data/db_info/' 
    
    print("--- Running Dataloader Verification for PanoMTDU ---")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"\nERROR: Please update DATASET_ROOT ({DATASET_ROOT}) to match your system.")
    else:
        # 1. Initialize the dataset (Train split)
        print("\n--- Initializing TRAIN split ---")
        # 注意：这里假设你的 __init__ 已经改回使用全局 DB_INFO_DIR 或默认路径逻辑
        # 如果你保留了 split_dir 参数，请在这里传入
        train_dataset = PanoMTDU(root=DATASET_ROOT, split='train')
        print(f"Dataset Loaded. Length: {len(train_dataset)}")
        
        # 2. Get a single sample for detailed checking
        print("\nFetching a sample from the TRAIN dataset (index 0)...")
        sample = train_dataset[0]
        
        # 3. Verify the top-level structure
        print("Sample keys:", sample.keys())
        assert 'pano' in sample and 'random' in sample and 'merged' in sample
        
        # --- 4. Verify Panorama Data Content ---
        print("\n--- Verifying PANO Data Shapes and Types ---")
        pano_data = sample['pano']
        pano_img = pano_data['image']
        
        print(f"Image:         shape={pano_img.shape}, dtype={pano_img.dtype}")
        print(f"Metadata:      {sample['meta']}")

        # --- 5. Verify Random View Data Content ---
        print("\n--- Verifying RANDOM View Data Shapes and Types ---")
        rand_data = sample['random']
        rand_sem = rand_data['semseg']
        rand_dep = rand_data['depth']
        rand_norm = rand_data['normals']

        # 计算统计量 (仿照 structured3d)
        valid_rand_dep = rand_dep[rand_dep > 0] # PanoMTDU depth > 0 is valid
        rand_dep_min = valid_rand_dep.min() if valid_rand_dep.size > 0 else 'N/A'
        rand_dep_max = valid_rand_dep.max() if valid_rand_dep.size > 0 else 'N/A'

        print(f"SemSeg:        shape={rand_sem.shape}, dtype={rand_sem.dtype}, unique_vals={np.unique(rand_sem).tolist()[:10]}...") 
        print(f"Depth:         shape={rand_dep.shape}, dtype={rand_dep.dtype}, min={rand_dep_min}, max={rand_dep_max}")
        print(f"Normals:       shape={rand_norm.shape}, dtype={rand_norm.dtype}, min={rand_norm.min():.3f}, max={rand_norm.max():.3f}")

        # --- 6. Verify Merged View Data Content ---
        print("\n--- Verifying MERGED View Data Shapes and Types ---")
        merged_data = sample['merged']
        merged_sem = merged_data['semseg']
        merged_dep = merged_data['depth']
        merged_norm = merged_data['normals']
        merged_conf = merged_data['conf']

        valid_merged_dep = merged_dep[merged_dep > 0]
        merged_dep_min = valid_merged_dep.min() if valid_merged_dep.size > 0 else 'N/A'
        merged_dep_max = valid_merged_dep.max() if valid_merged_dep.size > 0 else 'N/A'

        print(f"SemSeg:        shape={merged_sem.shape}, dtype={merged_sem.dtype}, unique_vals={np.unique(merged_sem).tolist()[:10]}...")
        print(f"Depth:         shape={merged_dep.shape}, dtype={merged_dep.dtype}, min={merged_dep_min}, max={merged_dep_max}")
        print(f"Normals:       shape={merged_norm.shape}, dtype={merged_norm.dtype}, min={merged_norm.min():.3f}, max={merged_norm.max():.3f}")
        print(f"Confidence:    shape={merged_conf.shape}, dtype={merged_conf.dtype}, min={merged_conf.min():.3f}, max={merged_conf.max():.3f}")

        # --- 7. Specific Check for Ignore Index (SemSeg) ---
        print("\n--- Verifying Ignore Index (SemSeg) ---")
        # 你的要求：检查 255 是否存在
        sem_unique = np.unique(merged_sem)
        if 255 in sem_unique:
            print(f"Confirmed: 255 is present as ignore index in Merged SemSeg. (Unique vals count: {len(sem_unique)})")
        else:
            print(f"Note: 255 not found in this specific sample's Merged SemSeg (might be fully valid).")
            
        # 检查 Random View 的 Ignore Index
        rand_sem_unique = np.unique(rand_sem)
        if 255 in rand_sem_unique:
             print(f"Confirmed: 255 is present as ignore index in Random SemSeg.")

        print("\nVerification complete. The PanoMTDU dataloader is ready.")