import os
import glob
import numpy as np
from PIL import Image
import torch.utils.data as data

# 统一配置常量
H, W = (512, 1024)  # 固定的全景图尺寸 (高度, 宽度)

class Deep360_Depth(data.Dataset):
    """
    Deep360 dataset for Monocular Depth Estimation.
    Strictly aligns the camera 1 RGB view (_12_rgb1) with the ground truth Depth map.
    """

    def __init__(self, root, split='training', transform=None, retname=True):
        """
        Args:
            root (str): 处理后的 Deep360 根目录 (如 '/mnt/localssd/Deep360_Processed').
            split (str or list): 数据集划分 ('training', 'validation', 'testing').
            transform (callable, optional): 数据增强操作.
            retname (bool): 是否返回 metadata.
        """
        self.root = root
        self.transform = transform
        self.retname = retname
        
        if isinstance(split, str):
            self.split = [split]
        else:
            self.split = split

        self.data_list = []
        print(f"Initializing dataloader for Deep360 {'/'.join(self.split)} set(s)...")
        
        # 遍历所有的 episode 文件夹 (ep1_500frames ~ ep6_500frames)
        ep_dirs = glob.glob(os.path.join(self.root, "ep*"))
        
        for ep_dir in ep_dirs:
            ep_name = os.path.basename(ep_dir)
            for splt in self.split:
                depth_dir = os.path.join(ep_dir, splt, 'depth')
                rgb_dir = os.path.join(ep_dir, splt, 'rgb')
                
                if not os.path.exists(depth_dir) or not os.path.exists(rgb_dir):
                    continue
                
                # 1. 永远以唯一的 GT Depth 为锚点进行遍历
                depth_files = glob.glob(os.path.join(depth_dir, "*_depth.png"))
                
                for depth_path in depth_files:
                    # 提取帧号: '004008_depth.png' -> '004008'
                    frame_id = os.path.basename(depth_path).split('_')[0]
                    
                    # 2. 核心铁律：强制只读取 12_rgb1，丢弃其他 11 张错位视角的图片
                    rgb_name = f"{frame_id}_12_rgb1.png"
                    rgb_path = os.path.join(rgb_dir, rgb_name)
                    
                    # 3. 只有当对应的正确视角 RGB 存在时，才加入训练列表
                    if os.path.exists(rgb_path):
                        self.data_list.append({
                            'img_path': rgb_path,
                            'depth_path': depth_path,
                            'ep_name': ep_name,
                            'split': splt,
                            'frame_id': frame_id
                        })

        if not self.data_list:
            raise RuntimeError(f"未找到匹配的 Deep360 数据: split={self.split}")

        print(f"Initialized Deep360_Depth Dataset. Found {len(self.data_list)} valid strictly-aligned samples.")

    def __getitem__(self, index):
        item_info = self.data_list[index]
        
        sample = {
            'image': self._load_img(item_info['img_path']),
            'depth': self._load_depth(item_info['depth_path'])
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # 嵌套为 'pano' 以保持多任务基类的字典格式统一
        final_sample = {'pano': sample}

        if self.retname:
            final_sample['meta'] = {
                'ep_name': item_info['ep_name'],
                'split': item_info['split'],
                'frame_id': item_info['frame_id'],
                'img_size': (sample['image'].shape[0], sample['image'].shape[1])
            }

        return final_sample

    def __len__(self):
        return len(self.data_list)

    def _load_img(self, path):
        _img = Image.open(path).convert('RGB')
        _img = _img.resize((W, H), Image.BILINEAR)
        return np.array(_img, dtype=np.float32)

    def _load_depth(self, path):
        if not os.path.exists(path):
            return np.zeros((H, W, 1), dtype=np.float32)
            
        _depth = Image.open(path)
        _depth = _depth.resize((W, H), Image.NEAREST)
        _depth_16bit = np.array(_depth, dtype=np.uint16)
        _depth_m = _depth_16bit.astype(np.float32) / 650.0
        _depth_m[_depth_m > 99.0] = 0.0
        
        return np.expand_dims(_depth_m, axis=2)

# ==========================================
# 测试主函数 (Main Test Block)
# ==========================================
if __name__ == '__main__':
    # 配置为你处理后的 Deep360 数据根目录
    DATASET_ROOT = '/mnt/localssd/Deep360_Processed'
    
    print("--- 启动 Deep360_Depth Dataloader 测试 ---")
    
    if not os.path.exists(DATASET_ROOT):
        print(f"\n[错误] 请确认 DATASET_ROOT ({DATASET_ROOT}) 路径是否正确。")
    else:
        try:
            # 1. 实例化 Dataset (测试 testing split，里面有 100 张 GT)
            val_dataset = Deep360_Depth(root=DATASET_ROOT, split='testing')
            
            # 2. 随机获取一个样本
            print("\n[1/3] 正在加载样本 (Index: 0) ...")
            sample = val_dataset[0]
            
            # 3. 验证字典结构
            assert 'pano' in sample, "样本缺少 'pano' 键值"
            assert 'meta' in sample, "样本缺少 'meta' 键值"
            
            pano_data = sample['pano']
            img = pano_data['image']
            dep = pano_data['depth']
            
            # 4. 打印数据详细信息
            print("\n[2/3] 样本 Metadata:")
            print(f"  - Episode:      {sample['meta']['ep_name']}")
            print(f"  - Split:        {sample['meta']['split']}")
            print(f"  - Frame ID:     {sample['meta']['frame_id']}")
            print(f"  - Image Size:   {sample['meta']['img_size']}")

            print("\n[3/3] 数据张量 (Shapes & Types & Values):")
            print(f"  - [RGB]      Shape: {img.shape}, Dtype: {img.dtype}")
            
            # 统计深度有效范围
            valid_dep = dep[dep > 0]
            dep_min = valid_dep.min() if valid_dep.size > 0 else 0
            dep_max = valid_dep.max() if valid_dep.size > 0 else 0
            print(f"  - [Depth]    Shape: {dep.shape}, Dtype: {dep.dtype}")
            print(f"               有效物理深度范围 (米): Min={dep_min:.3f}m, Max={dep_max:.3f}m")
            
            print("\n🎉 测试通过！Deep360 严格对齐版 Dataloader 运行正常。")
            
        except Exception as e:
            print(f"\n[测试失败] 运行中出现错误: {e}")