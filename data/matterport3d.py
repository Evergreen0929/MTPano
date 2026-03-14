import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data

# 统一配置 JSON 所在目录
DB_INFO_DIR = '/mnt/localssd/code/PanoMTL/data/db_info/'
H, W = (512, 1024)  # 固定的全景图尺寸 (高度, 宽度)

class Matterport3D_MT(data.Dataset):
    """
    Matterport3D dataset for multi-task learning.
    Strictly follows the official 61/11/18 train/val/test scene splits via JSON files.
    """

    def __init__(self, root, split='train', transform=None, retname=True):
        """
        Args:
            root (str): 数据集的根目录，即包含 'mp3d_aligned_labels' 和 'mp3d_results_panoramas' 的文件夹.
            split (str or list): 数据集划分 ('train', 'val', 'test').
            transform (callable, optional): 适用于全景图和标签的数据增强操作.
            retname (bool): 是否返回 metadata.
        """
        self.root = root
        self.transform = transform
        self.retname = retname
        self.pano_dir = os.path.join(self.root, 'mp3d_results_panoramas')
        self.label_dir = os.path.join(self.root, 'mp3d_aligned_labels')
        
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.data_list = []
        print(f"Initializing dataloader for Matterport3D {'/'.join(self.split)} set(s)...")
        
        # 核心修改：通过 JSON 文件加载官方划分数据
        for splt in self.split:
            json_file = os.path.join(DB_INFO_DIR, f'matterport3d_pairs_{splt}.json')
            try:
                with open(json_file, 'r') as f:
                    self.data_list.extend(json.load(f))
            except FileNotFoundError:
                raise RuntimeError(f"找不到 JSON 划分文件: {json_file}。请先运行 json 生成脚本！")
        
        if not self.data_list:
            raise RuntimeError(f"JSON 中没有数据: {self.split}")

        print(f"Initialized Matterport3D_MT Dataset. Found {len(self.data_list)} samples for {'/'.join(self.split)}.")

    def __getitem__(self, index):
        item_info = self.data_list[index]
        scan_id = item_info['scan_id']
        vp_id = item_info['viewpoint_id']
        
        img_path = os.path.join(self.pano_dir, scan_id, f"{vp_id}_panorama.jpg")
        sem_path = os.path.join(self.label_dir, scan_id, 'semantic', f"{vp_id}_semantic.png")
        dep_path = os.path.join(self.label_dir, scan_id, 'depth', f"{vp_id}_depth.png")
        norm_path = os.path.join(self.label_dir, scan_id, 'normal', f"{vp_id}_normal.png")
        
        sample = {
            'image': self._load_img(img_path),
            'semseg': self._load_semseg(sem_path),
            'depth': self._load_depth(dep_path),
            'normals': self._load_normals(norm_path)
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # 嵌套一层 'pano' 是为了保持和 structured3d.py / pano_mtdu.py 统一的字典结构
        final_sample = {'pano': sample}

        if self.retname:
            final_sample['meta'] = {
                'scan_id': scan_id,
                'viewpoint_id': vp_id,
                'img_size': (sample['image'].shape[0], sample['image'].shape[1])
            }

        return final_sample

    def __len__(self):
        return len(self.data_list)

    def _load_img(self, path):
        _img = Image.open(path).convert('RGB')
        _img = _img.resize((W, H), Image.BILINEAR)
        return np.array(_img, dtype=np.float32)

    def _load_semseg(self, path):
        if not os.path.exists(path):
            return np.full((H, W, 1), 255, dtype=np.uint8)
        _semseg = Image.open(path)
        _semseg = _semseg.resize((W, H), Image.NEAREST)
        _semseg = np.array(_semseg, dtype=np.uint8)
        _semseg = (np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1).astype(np.uint8)
        _semseg[_semseg == -1] = 255 
        return _semseg

    def _load_depth(self, path):
        if not os.path.exists(path):
            return np.zeros((H, W, 1), dtype=np.float32)
        _depth = Image.open(path)
        _depth = _depth.resize((W, H), Image.NEAREST)
        _depth_mm = np.array(_depth, dtype=np.uint16)
        _depth_m = _depth_mm.astype(np.float32) / 1000.0
        _depth_m[_depth_m > 10.0] = 0.0
        return np.expand_dims(_depth_m, axis=2)

    def _load_normals(self, path):
        if not os.path.exists(path):
            return np.zeros((H, W, 3), dtype=np.float32)
        _normals_img = Image.open(path).convert('RGB')
        _normals_img = _normals_img.resize((W, H), Image.NEAREST)
        _normals_arr = np.array(_normals_img, dtype=np.float32)
        mask_invalid = (_normals_arr == 128).all(axis=2)
        _normals = (_normals_arr / 255.0) * 2.0 - 1.0
        _normals[mask_invalid] = 0
        return _normals

# ==========================================
# 测试主函数 (Main Test Block)
# ==========================================
if __name__ == '__main__':
    # 配置你的数据根目录
    DATASET_ROOT = '/mnt/localssd/Matterport3D_Processed'
    
    print("--- 启动 Matterport3D_MT Dataloader 测试 ---")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"\\n[错误] 请确认 DATASET_ROOT ({DATASET_ROOT}) 路径是否正确。")
    elif not os.path.exists(DB_INFO_DIR):
        print(f"\\n[错误] 请确认 DB_INFO_DIR ({DB_INFO_DIR}) 路径是否正确。")
    else:
        try:
            # 1. 实例化 Dataset (默认加载 train split)
            # 注意：这需要你已经运行过生成 json 的脚本
            train_dataset = Matterport3D_MT(root=DATASET_ROOT, split='train')
            
            # 2. 随机获取一个样本 (这里取 index=0)
            print("\\n[1/3] 正在加载样本 (Index: 0) ...")
            sample = train_dataset[0]
            
            # 3. 验证字典结构
            assert 'pano' in sample, "样本缺少 'pano' 键值"
            assert 'meta' in sample, "样本缺少 'meta' 键值"
            
            pano_data = sample['pano']
            img = pano_data['image']
            sem = pano_data['semseg']
            dep = pano_data['depth']
            norm = pano_data['normals']
            
            # 4. 打印数据详细信息
            print("\\n[2/3] 样本 Metadata:")
            print(f"  - Scan ID:      {sample['meta']['scan_id']}")
            print(f"  - Viewpoint ID: {sample['meta']['viewpoint_id']}")
            print(f"  - Image Size:   {sample['meta']['img_size']}")

            print("\\n[3/3] 数据张量 (Shapes & Types & Values):")
            print(f"  - [RGB]      Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"  - [Semantic] Shape: {sem.shape}, Dtype: {sem.dtype}")
            print(f"               包含的类别 IDs: {np.unique(sem).tolist()[:10]} ...")
            
            valid_dep = dep[dep > 0]
            dep_min = valid_dep.min() if valid_dep.size > 0 else 0
            dep_max = valid_dep.max() if valid_dep.size > 0 else 0
            print(f"  - [Depth]    Shape: {dep.shape}, Dtype: {dep.dtype}")
            print(f"               有效深度范围 (米): Min={dep_min:.3f}, Max={dep_max:.3f}")
            
            print(f"  - [Normals]  Shape: {norm.shape}, Dtype: {norm.dtype}")
            print(f"               向量范围: Min={norm.min():.3f}, Max={norm.max():.3f}")
            
            # 5. 特殊状态验证
            print("\\n--- 特殊状态验证 ---")
            if 255 in np.unique(sem):
                print("✅ 语义分割 (SemSeg) 成功保留了 Ignore Index (255)。")
            else:
                print("⚠️ 语义分割中未发现 Ignore Index (255)，请确认该全景图是否完全被有效物体填满。")
                
            if np.sum(dep == 0) > 0:
                print("✅ 深度图 (Depth) 包含无效区域 (值为 0)。")
            
            print("\\n🎉 测试通过！Dataloader 运行正常。")
            
        except Exception as e:
            print(f"\\n[测试失败] 运行中出现错误: {e}")