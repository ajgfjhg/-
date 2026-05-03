import os
import random
import numpy as np
from collections import defaultdict
from torch.utils import data
from PIL import Image
from torchvision import transforms
from random import shuffle, choice

class RoadTagProducer:
    class ImagePackage:
        def __init__(self, image_path, image_label):
            self.image_path = image_path
            self.image_label = image_label

    class ImageDataset(data.Dataset):
        def __init__(self, image_package_list, transform=None):
            self.image_package_list = image_package_list
            self.transform = transform

        def __len__(self):
            return len(self.image_package_list)

        def __getitem__(self, index):
            package = self.image_package_list[index]
            # 增加异常处理，防止个别损坏图片导致训练中断
            try:
                img = Image.open(package.image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {package.image_path}: {e}")
                # 返回一个全黑图作为占位（或者根据需要处理）
                img = Image.new('RGB', (224, 224))

            if self.transform:
                img = self.transform(img)
            return img, package.image_label, package.image_path

    def __init__(self, train_dir_path, test_dir_path, batch_size=128, test_remain=0.1, seed=42, mean=1500):
        random.seed(seed)
        np.random.seed(seed)

        # 1. 标签映射
        all_dirs = sorted([d for d in os.listdir(train_dir_path) if os.path.isdir(os.path.join(train_dir_path, d))])
        self.label_map = {dir_name: i for i, dir_name in enumerate(all_dirs)}
        print(f"--- 类别映射完成: 检测到 {len(all_dirs)} 个类别 ---")

        # 2. 预处理
        self.train_transform = transforms.Compose([
            transforms.Resize([232, 232]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 增加光照增强，防止过拟合
            transforms.RandomCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize([256, 256]), # 先缩放到稍大
            transforms.CenterCrop([224, 224]), # 中心裁剪，保证物体形状不被压缩
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. 分类加载原始路径（不凑数）
        train_packages = []
        val_packages = []
        
        for dir_name in all_dirs:
            label = self.label_map[dir_name]
            dir_full_path = os.path.join(train_dir_path, dir_name)
            if not os.path.exists(dir_full_path): continue
            
            # 获取该类下所有原始图片
            images = [os.path.join(dir_full_path, f) for f in os.listdir(dir_full_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: continue
            
            # 【关键步骤】先对原始图片进行物理切分
            shuffle(images)
            split_idx = int(len(images) * (1 - test_remain))
            raw_train = images[:split_idx]
            raw_val = images[split_idx:]

            # --- 对训练集部分进行过采样（凑数） ---
            final_train = list(raw_train)
            if len(final_train) > mean:
                final_train = final_train[:mean]
            else:
                while len(final_train) < mean:
                    final_train.append(choice(raw_train)) # 从当前类的训练部分选

            # 封装成 ImagePackage
            for p in final_train:
                train_packages.append(self.ImagePackage(p, label))
            for p in raw_val:
                val_packages.append(self.ImagePackage(p, label))

        # 4. 加载独立测试集
        self.test_images = []
        for dir_name in all_dirs:
            label = self.label_map.get(dir_name)
            if label is None: continue
            dir_full_path = os.path.join(test_dir_path, dir_name)
            if not os.path.exists(dir_full_path): continue
            
            images = [os.path.join(dir_full_path, f) for f in os.listdir(dir_full_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_path in images:
                self.test_images.append(self.ImagePackage(img_path, label))

        # 5. 定义 DataLoader
        self.train_data_loader = data.DataLoader(
            self.ImageDataset(train_packages, self.train_transform),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.train_test_data_loader = data.DataLoader(
            self.ImageDataset(val_packages, self.test_transform),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        self.test_data_loader = data.DataLoader(
            self.ImageDataset(self.test_images, self.test_transform),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        print(f"--- 数据准备完成: 训练集 {len(train_packages)} 张, 验证集 {len(val_packages)} 张, 测试集 {len(self.test_images)} 张 ---")