import os
import glob
import numpy as np
from PIL import Image
import torch.utils.data as data

# 统一配置常量
H, W = (512, 1024)  # 固定的全景图尺寸 (高度, 宽度)

class SynPASS_Seg(data.Dataset):
    """
    SynPASS dataset for Semantic Segmentation.
    Follows the official train/val/test splits and handles varying weather conditions.
    """

    def __init__(self, root, split='train', weather='all', transform=None, retname=True):
        """
        Args:
            root (str): 数据集的根目录，包含 'img' 和 'semantic' 的文件夹.
            split (str or list): 数据集划分 ('train', 'val', 'test').
            weather (str or list): 天气条件 ('sun', 'cloud', 'fog', 'rain', 'all').
            transform (callable, optional): 适用于全景图和标签的数据增强操作.
            retname (bool): 是否返回 metadata.
        """
        self.root = root
        self.transform = transform
        self.retname = retname
        self.img_dir = os.path.join(self.root, 'img')
        self.label_dir = os.path.join(self.root, 'semantic')
        
        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split

        if isinstance(weather, str):
            if weather == 'all':
                self.weather = ['sun', 'cloud', 'fog', 'rain']
            else:
                self.weather = [weather]
        else:
             self.weather = weather

        self.data_list = []
        print(f"Initializing dataloader for SynPASS {'/'.join(self.split)} set(s) under {','.join(self.weather)} weather...")
        
        # 核心：遍历指定 weather 和 split 下的所有图片
        for w_cond in self.weather:
            for splt in self.split:
                search_pattern = os.path.join(self.img_dir, w_cond, splt, '*', '*.jpg')
                img_files = glob.glob(search_pattern)
                
                for img_path in img_files:
                    # 获取相对路径以便找到对应的 label
                    # 例如: MAP_1_point14/000000.jpg
                    rel_path = os.path.relpath(img_path, os.path.join(self.img_dir, w_cond, splt))
                    
                    # 构建对应的 semantic label 路径
                    sem_path = os.path.join(self.label_dir, w_cond, splt, rel_path).replace('.jpg', '_trainID.png')
                    
                    # 提取 metadata 信息
                    map_point_dir = os.path.dirname(rel_path)
                    img_name = os.path.basename(rel_path)
                    
                    self.data_list.append({
                        'img_path': img_path,
                        'sem_path': sem_path,
                        'weather': w_cond,
                        'split': splt,
                        'map_point': map_point_dir,
                        'img_name': img_name
                    })

        if not self.data_list:
            raise RuntimeError(f"未找到匹配的数据: split={self.split}, weather={self.weather}")

        print(f"Initialized SynPASS_Seg Dataset. Found {len(self.data_list)} samples.")

    def __getitem__(self, index):
        item_info = self.data_list[index]
        
        sample = {
            'image': self._load_img(item_info['img_path']),
            'semseg': self._load_semseg(item_info['sem_path'])
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # 嵌套一层 'pano' 是为了保持和 structured3d.py / pano_mtdu.py 统一的字典结构
        final_sample = {'pano': sample}

        if self.retname:
            final_sample['meta'] = {
                'weather': item_info['weather'],
                'split': item_info['split'],
                'map_point': item_info['map_point'],
                'img_name': item_info['img_name'],
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
            print(f"Warning: Label not found at {path}. Returning empty mask.")
            return np.full((H, W, 1), 255, dtype=np.uint8)
        
        _semseg = Image.open(path)
        _semseg = _semseg.resize((W, H), Image.NEAREST)
        _semseg = np.array(_semseg, dtype=np.uint8)
        _semseg = (np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1).astype(np.uint8)
        _semseg[_semseg == -1] = 255
        return _semseg

# ==========================================
# 测试主函数 (Main Test Block)
# ==========================================
if __name__ == '__main__':
    # 配置你的数据根目录
    DATASET_ROOT = '/mnt/localssd/SynPASS'
    
    print("--- 启动 SynPASS_Seg Dataloader 测试 ---")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"\n[错误] 请确认 DATASET_ROOT ({DATASET_ROOT}) 路径是否正确。")
    else:
        try:
            # 1. 实例化 Dataset (加载 sun 天气下的 train split)
            train_dataset = SynPASS_Seg(root=DATASET_ROOT, split='train', weather='rain')
            
            # 2. 随机获取一个样本 (这里取 index=0)
            print("\n[1/3] 正在加载样本 (Index: 0) ...")
            sample = train_dataset[0]
            
            # 3. 验证字典结构
            assert 'pano' in sample, "样本缺少 'pano' 键值"
            assert 'meta' in sample, "样本缺少 'meta' 键值"
            
            pano_data = sample['pano']
            img = pano_data['image']
            sem = pano_data['semseg']
            
            # 4. 打印数据详细信息
            print("\n[2/3] 样本 Metadata:")
            print(f"  - Weather:      {sample['meta']['weather']}")
            print(f"  - Split:        {sample['meta']['split']}")
            print(f"  - Map Point:    {sample['meta']['map_point']}")
            print(f"  - Image Name:   {sample['meta']['img_name']}")
            print(f"  - Image Size:   {sample['meta']['img_size']}")

            print("\n[3/3] 数据张量 (Shapes & Types & Values):")
            print(f"  - [RGB]      Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"  - [Semantic] Shape: {sem.shape}, Dtype: {sem.dtype}")
            
            unique_ids = np.unique(sem).tolist()
            print(f"               包含的类别 IDs: {unique_ids}")
            
            # 5. 特殊状态验证
            print("\n--- 特殊状态验证 ---")
            if 255 in unique_ids:
                print("✅ 语义分割 (SemSeg) 包含 Ignore Index (255)。")
            else:
                print("ℹ️ 语义分割中未发现 Ignore Index (255)，这在全覆盖的合成数据集中是正常的。")
                
            print("\n🎉 测试通过！Dataloader 运行正常。")
            
        except Exception as e:
            print(f"\n[测试失败] 运行中出现错误: {e}")