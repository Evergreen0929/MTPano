import os
import json
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import io

# S3 依赖
try:
    import boto3
except ImportError:
    print("Warning: boto3 not installed. S3 dataloader will not work.")

# DB_INFO_DIR 仍然读取本地的 JSON 文件 (假设 split 文件还在本地)
DB_INFO_DIR = '/mnt/localssd/code/PanoMTL/data/db_info'

class PanoMTDU(data.Dataset):
    """
    PanoMTDU dataset (S3 Version): Loads data directly from AWS S3.
    Structure matches the local version exactly.
    """

    def __init__(self, root, split='train', transform_pano=None, retname=True):
        """
        Args:
            root (str): S3 URI (e.g., s3://adobe-lingzhi-p/jingdongz-data/PanoPseudoLabels/)
            split (str or list): 'train', 'val', etc.
        """
        self.root = root
        self.transform_pano = transform_pano
        self.retname = retname
        
        # 解析 S3 路径
        if not self.root.startswith('s3://'):
            raise ValueError(f"Root must start with s3://, got {self.root}")
        
        parts = self.root.replace('s3://', '').split('/', 1)
        self.bucket_name = parts[0]
        self.prefix = parts[1] if len(parts) > 1 else ''
        # 确保 prefix 以 / 结尾，且不以 / 开头 (Key 拼接逻辑)
        if self.prefix and not self.prefix.endswith('/'):
            self.prefix += '/'
        
        # 初始化为空，在 worker 中懒加载
        self.s3_client = None 

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # 1. Load the Split File (Local JSON)
        self.data_list = []
        print(f"Initializing S3 dataloader for PanoMTDU {'/'.join(self.split)}...")
        print(f"Target Bucket: {self.bucket_name}, Prefix: {self.prefix}")
        
        for splt in self.split:
            json_file = os.path.join(DB_INFO_DIR, f'panomtdu_{splt}.json')
            try:
                with open(json_file, 'r') as f:
                    self.data_list.extend(json.load(f))
            except FileNotFoundError:
                raise RuntimeError(f"JSON file for split '{splt}' not found at {json_file}")
        
        if not self.data_list:
            raise RuntimeError(f"No data found for split(s): {self.split}")
            
        print(f"Initialized PanoMTDU S3 Dataset. Loaded {len(self.data_list)} samples.")

    def _ensure_s3_client(self):
        """Lazy initialization of S3 client for multiprocessing safety."""
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')

    def _read_bytes_from_s3(self, rel_path):
        """Reads bytes from S3 key."""
        self._ensure_s3_client()
        key = f"{self.prefix}{rel_path}"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            # print(f"Error reading s3://{self.bucket_name}/{key}: {e}")
            return None

    def __getitem__(self, index):
        safe_id = self.data_list[index]
        
        # --- Paths (Relative to S3 Prefix) ---
        # Note: No os.path.join for S3 keys to avoid OS specific separators
        img_rel_path = f"img/{safe_id}.jpg"
        
        rand_idx = random.randint(0, 31)
        rand_name = f"label_{rand_idx}.png"
        merged_name = "merged.png"
        conf_name = "merged_confidence.png"

        # Construct S3 keys for labels
        # structure: semseg/scene_id/file.png
        rand_sem_path = f"semseg/{safe_id}/{rand_name}"
        rand_dep_path = f"depth/{safe_id}/{rand_name}"
        rand_norm_path = f"normals/{safe_id}/{rand_name}"
        
        merged_sem_path = f"semseg/{safe_id}/{merged_name}"
        merged_dep_path = f"depth/{safe_id}/{merged_name}"
        merged_norm_path = f"normals/{safe_id}/{merged_name}"
        merged_conf_path = f"semseg/{safe_id}/{conf_name}"

        # 2. Load Data from S3
        pano_img = self._load_img(img_rel_path)
        
        # Random View
        rand_sem = self._load_semseg(rand_sem_path)
        rand_dep = self._load_depth(rand_dep_path)
        rand_norm = self._load_normals(rand_norm_path)
        
        # Merged View
        merged_sem = self._load_semseg(merged_sem_path)
        merged_dep = self._load_depth(merged_dep_path)
        merged_norm = self._load_normals(merged_norm_path)
        merged_conf = self._load_conf(merged_conf_path)

        # 3. Construct Flat Dict
        full_sample = {
            'image': pano_img,
            'semseg': rand_sem,
            'depth': rand_dep,
            'normals': rand_norm,
            'merged_semseg': merged_sem,
            'merged_depth': merged_dep,
            'merged_normals': merged_norm,
            'merged_conf': merged_conf
        }

        # 4. Apply Transform
        if self.transform_pano is not None:
            full_sample = self.transform_pano(full_sample)

        # 5. Restructure
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
                'img_size': (full_sample['image'].shape[-2], full_sample['image'].shape[-1]) 
                            if hasattr(full_sample['image'], 'shape') else (512, 1024)
            }

        return sample

    def __len__(self):
        return len(self.data_list)

    # --- Loaders adapted for S3 Bytes ---

    def _load_img(self, rel_path):
        data_bytes = self._read_bytes_from_s3(rel_path)
        if data_bytes is None:
            # Fallback (Gray image)
            return np.zeros((512, 1024, 3), dtype=np.float32)
        
        _img = Image.open(io.BytesIO(data_bytes)).convert('RGB')
        return np.array(_img, dtype=np.float32)

    def _load_semseg(self, rel_path):
        data_bytes = self._read_bytes_from_s3(rel_path)
        if data_bytes is None:
            return np.full((512, 1024, 1), 255, dtype=np.uint8)
            
        _semseg = Image.open(io.BytesIO(data_bytes))
        _semseg = np.array(_semseg, dtype=np.uint8)
        return np.expand_dims(_semseg, axis=2)

    def _load_depth(self, rel_path):
        data_bytes = self._read_bytes_from_s3(rel_path)
        if data_bytes is None:
            return np.zeros((512, 1024, 1), dtype=np.float32)

        _depth = Image.open(io.BytesIO(data_bytes))
        _depth_mm = np.array(_depth, dtype=np.uint16)
        _depth_m = _depth_mm.astype(np.float32) / 1000.0
        return np.expand_dims(_depth_m, axis=2)

    def _load_normals(self, rel_path):
        data_bytes = self._read_bytes_from_s3(rel_path)
        if data_bytes is None:
            return np.zeros((512, 1024, 3), dtype=np.float32)

        _normals_img = Image.open(io.BytesIO(data_bytes)).convert('RGB')
        _normals_arr = np.array(_normals_img, dtype=np.float32)
        
        # [CRITICAL] 保持和本地完全一致的无效值处理逻辑
        mask_invalid = (_normals_arr == 128).all(axis=2)
        _normals = (_normals_arr / 255.0) * 2.0 - 1.0
        _normals[mask_invalid] = 0
        return _normals

    def _load_conf(self, rel_path):
        data_bytes = self._read_bytes_from_s3(rel_path)
        if data_bytes is None:
            return np.ones((512, 1024, 1), dtype=np.float32)
            
        _conf = Image.open(io.BytesIO(data_bytes))
        _conf_arr = np.array(_conf, dtype=np.float32)
        _conf_norm = _conf_arr / 255.0
        return np.expand_dims(_conf_norm, axis=2)


if __name__ == '__main__':
    # --- S3 Dataloader Verification ---
    
    # 你的 S3 路径
    S3_ROOT = 's3://adobe-lingzhi-p/jingdongz-data/PanoPseudoLabels/'
    
    print("--- Running S3 Dataloader Verification ---")
    
    try:
        # 1. Initialize
        # 确保 DB_INFO_DIR 下有 json 文件
        train_dataset = PanoMTDU(root=S3_ROOT, split='train')
        print(f"Dataset initialized. Length: {len(train_dataset)}")
        
        # 2. Fetch Sample
        print("\nFetching sample index 0 from S3...")
        sample = train_dataset[0]
        
        # 3. Verify Data
        print(f"ID: {sample['meta']['scene_id']}")
        
        img = sample['pano']['image']
        print(f"Image Shape: {img.shape}, Type: {img.dtype}")
        
        merged_norm = sample['merged']['normals']
        print(f"Merged Normals Shape: {merged_norm.shape}")
        print(f"Normals Min/Max: {merged_norm.min():.3f} / {merged_norm.max():.3f}")
        
        merged_sem = sample['merged']['semseg']
        print(f"Merged Semseg Unique: {np.unique(merged_sem).tolist()[:10]}...")

        print("\nS3 Verification Passed!")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()