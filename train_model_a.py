import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models 
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# 显存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 数据加载器
from producer import RoadTagProducer
# 导入模型接口
from Swin import swin_tiny 
from ConvNeXtV2 import convnext_v2_tiny
from ViT import vit_tiny 

# ================== 超参数 ==================
BATCH_SIZE = 64   
EPOCH = 20        # 预训练微调，20轮通常效果最好
LEARNING_RATE = 0.00005
N_CLASS = 29
TEST_REMAIN = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================== 官方模型适配器 ==================
def get_resnet_pretrained(model_type, num_class):
    weights_dict = {
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
        'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
        'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,
    }
    m = getattr(models, model_type)(weights=weights_dict[model_type])
    m.fc = nn.Linear(m.fc.in_features, num_class)
    return m

def get_vgg19_pretrained(num_class):
    m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_class)
    return m

# ================== 核心训练函数 ==================
def train_model(model_builder, model_name, dataset):
    torch.cuda.empty_cache()
    print(f"\n{'='*40}\n开始训练: {model_name}\n{'='*40}")

    model = model_builder(N_CLASS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH)
    scaler = GradScaler()

    history = {"epoch":[], "train_loss":[], "val_acc":[], "val_precision":[], "val_recall":[], "val_f1":[]}
    max_acc = 0.0

    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0.0
        for train_data, train_label, _ in dataset.train_data_loader:
            train_data, train_label = train_data.to(device), train_label.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(train_data)
                loss = loss_function(pred, train_label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # 验证逻辑
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for test_data, test_label, _ in dataset.train_test_data_loader:
                test_data = test_data.to(device)
                with autocast():
                    pred = model(test_data)
                all_preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
                all_labels.extend(test_label.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = epoch_loss / len(dataset.train_data_loader)
        print(f"Epoch [{epoch+1}/{EPOCH}] Loss:{avg_loss:.4f} Acc:{acc:.2%} F1:{f1:.2%}")

        # 记录数据
        for k, v in zip(history.keys(), [epoch+1, avg_loss, acc, precision, recall, f1]):
            history[k].append(v)

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
        scheduler.step()

    pd.DataFrame(history).to_csv(f"{model_name}_log.csv", index=False)
    
    # 释放显存
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return max_acc

if __name__ == '__main__':
    train_path, test_path = r'Mushroom_dataset/train', r'Mushroom_dataset/test'
    dataset = RoadTagProducer(train_path, test_path, BATCH_SIZE, TEST_REMAIN)

    model_configs = [
        #('resnet18', lambda n: get_resnet_pretrained('resnet18', n)),
        #('resnet34', lambda n: get_resnet_pretrained('resnet34', n)),
        #('resnet50', lambda n: get_resnet_pretrained('resnet50', n)),
        #('resnet101', lambda n: get_resnet_pretrained('resnet101', n)),
        #('resnet152', lambda n: get_resnet_pretrained('resnet152', n)),
        ('vgg19', get_vgg19_pretrained),
        ('vit_tiny', vit_tiny),
        ('swin_tiny', swin_tiny),
        ('convnext_v2', convnext_v2_tiny),
    ]

    for name, builder in model_configs:
        train_model(builder, name, dataset)