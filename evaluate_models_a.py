import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import os
import timm

# ==================== 1. 导入你本地的模型定义 ====================
# 这样可以确保 get_model 创建的结构和你训练时完全一致
from ConvNeXtV2 import convnext_v2_tiny
from ViT import vit_tiny
from Swin import swin_tiny
# 如果 vgg19.py 里有定义，也可以从那里导入；如果没有，我们尝试 timm 的特定版本

# ==================== 配置参数 ====================
TEST_DATA_PATH = r"Mushroom_dataset/test"
NUM_CLASSES = 29
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 2. 修正后的模型适配工厂 ====================
# 在脚本开头确保导入了你自己的模型
from vgg19 import vgg19 as vgg19_local
from ViT import vit_tiny
from ConvNeXtV2 import convnext_v2_tiny
from Swin import swin_tiny

def get_model(model_name):
    if model_name.startswith("resnet"):
        model = getattr(models, model_name)(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    elif model_name == "vgg19":
        # 与训练时保持一致，使用 torchvision 的 vgg19
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        return model

    elif model_name == "convnext_v2_tiny":
        # 确保 ConvNeXtV2.py 中是 timm 版本
        from ConvNeXtV2 import convnext_v2_tiny
        return convnext_v2_tiny(num_class=NUM_CLASSES)

    elif model_name == "swin_tiny":
        from Swin import swin_tiny
        return swin_tiny(num_class=NUM_CLASSES)

    elif model_name == "vit_tiny":
        from ViT import vit_tiny
        return vit_tiny(num_class=NUM_CLASSES)

    else:
        raise ValueError(f"未知模型: {model_name}")

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

# ==================== 3. 主程序 (文件名精准对齐) ====================
if __name__ == '__main__':
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：测试集路径 {TEST_DATA_PATH} 不存在！")
        exit()

    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 【精准文件名匹配】
    models_info = {
        "resnet18": "resnet18_best.pth",
        "resnet34": "resnet34_best.pth",
        "resnet50": "resnet50_best.pth",
        "resnet101": "resnet101_best.pth",
        "resnet152": "resnet152_best.pth",
        "vgg19": "vgg19_best.pth",
        "convnext_v2_tiny": "convnext_v2_best.pth", # 修正文件名
        "swin_tiny": "swin_tiny_best.pth",
        "vit_tiny": "vit_tiny_best.pth"
    }

    results = []
    for name, weight_file in models_info.items():
        if not os.path.exists(weight_file):
            print(f"跳过：找不到 {weight_file}")
            continue
        
        print(f"\n---> 正在评估: {name}")
        try:
            model = get_model(name).to(DEVICE)
            state_dict = torch.load(weight_file, map_location=DEVICE)
            
            # 兼容性处理
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                if k.startswith('_orig_mod.'): k = k[10:]
                new_state_dict[k] = v

            # 加载权重 (使用 strict=False 增加容错性)
            model.load_state_dict(new_state_dict, strict=False)
            
            acc, prec, rec, f1 = evaluate(model, test_loader, DEVICE)
            results.append({
                "模型名称": name,
                "准确率": f"{acc*100:.2f}%",
                "F1分数": f"{f1*100:.2f}%"
            })
            print(f"结果: Acc={acc*100:.2f}%, F1={f1*100:.2f}%")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"评估 {name} 失败: {e}")

    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*20 + " 最终对比报告 " + "="*20)
        print(df.to_string(index=False))
        df.to_csv("mogu_results.csv", index=False, encoding='utf-8-sig')