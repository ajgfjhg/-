import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import os

# ==================== 导入自定义模型 ====================
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from vgg19 import vgg19
from ViT import vit_tiny
from ConvNeXtV2 import convnext_v2_tiny
from Swin import swin_tiny

# ==================== 配置参数 ====================
TEST_DATA_PATH = r"Mushroom_dataset/test"   # 独立测试集路径
NUM_CLASSES = 29
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试集预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 定义函数（放在全局作用域） ====================
def get_model(model_name):
    """根据模型名称返回模型实例"""
    if model_name == "vgg19":
        return vgg19(num_class=NUM_CLASSES)
    elif model_name == "resnet18":
        return resnet18(num_class=NUM_CLASSES)
    elif model_name == "resnet34":
        return resnet34(num_class=NUM_CLASSES)
    elif model_name == "resnet50":
        return resnet50(num_class=NUM_CLASSES)
    elif model_name == "resnet101":
        return resnet101(num_class=NUM_CLASSES)
    elif model_name == "resnet152":
        return resnet152(num_class=NUM_CLASSES)
    elif model_name == "convnext_v2_tiny":
        return convnext_v2_tiny(num_class=NUM_CLASSES)
    elif model_name == "swin_tiny":
        return swin_tiny(num_class=NUM_CLASSES)
    elif model_name == "vit_tiny":
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

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 1. 加载测试集
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：测试集路径 {TEST_DATA_PATH} 不存在！")
        exit()

    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
    # 在 Windows 下，num_workers > 0 必须在 if __name__ == '__main__': 之后
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"测试集样本数：{len(test_dataset)}，类别数：{len(test_dataset.classes)}")
    print(f"类别映射：{test_dataset.classes}")

    # 2. 模型信息
    models_info = {
        "vgg19": "vgg19_best.pth",
        "resnet18": "resnet18_best.pth",
        "resnet34": "resnet34_best.pth",
        "resnet50": "resnet50_best.pth",
        "resnet101": "resnet101_best.pth",
        "resnet152": "resnet152_best.pth",
        "convnext_v2_tiny": "convnext_v2_tiny_best.pth",
        "swin_tiny": "swin_tiny_best.pth",
        "vit_tiny": "vit_tiny_best.pth"
    }

    # 3. 循环评估
    results = []
    for name, weight_file in models_info.items():
        if not os.path.exists(weight_file):
            print(f"警告：权重文件 {weight_file} 不存在，跳过模型 {name}")
            continue
        
        print(f"\n正在处理模型: {name}")
        model = get_model(name).to(DEVICE)
        
        # 加载权重
        state_dict = torch.load(weight_file, map_location=DEVICE)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # 处理前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('_orig_mod.'):
                k = k[10:]
            new_state_dict[k] = v

        try:
            model.load_state_dict(new_state_dict, strict=True)
            acc, prec, rec, f1 = evaluate(model, test_loader, DEVICE)
            
            results.append({
                "模型名称": name,
                "准确率": f"{acc*100:.2f}%",
                "精确率": f"{prec*100:.2f}%",
                "召回率": f"{rec*100:.2f}%",
                "F1分数": f"{f1*100:.2f}%"
            })
            print(f"{name}: Acc={acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%")
        except Exception as e:
            print(f"加载模型 {name} 失败: {e}")

    # 4. 输出与保存
    if results:
        df = pd.DataFrame(results)
        print("\n========== 最终结果表格 ==========")
        print(df.to_string(index=False))
        df.to_csv("model_evaluation_results.csv", index=False, encoding='utf-8-sig')
        print("\n结果已保存到 model_evaluation_results.csv")
    else:
        print("\n没有可显示的评估结果。")