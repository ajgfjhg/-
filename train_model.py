import os
import gc
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

# 环境变量设置（显存优化）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"清空缓存后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

# 数据加载器
from producer import RoadTagProducer

# 模型定义
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from vgg19 import VGG
from ViT import vit_tiny
from ConvNeXtV2 import convnext_v2_tiny
from Swin import swin_tiny

# ================== 超参数 ==================
BATCH_SIZE = 128
EPOCH = 20
TEST_REMAIN = 0.1
LEARNING_RATE = 0.00005
N_CLASS = 29

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ================== 工具函数 ==================
def test_single_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor)
        pred_idx = torch.argmax(pred, dim=1).item()
    return pred_idx

def test_dir(model, dir_path, label_map=None):
    model.eval()
    if label_map is not None:
        folder_name = os.path.basename(dir_path)
        true_label = label_map[folder_name]
    else:
        true_label = int(dir_path[-5:])

    sum_acc, sum_count = 0, 0
    total_test_time = 0.0
    test_start_time = time.time()
    with torch.no_grad():
        for root, _, files in os.walk(dir_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image_path = os.path.join(root, file)
                img_start = time.time()
                result = test_single_image(model, image_path)
                total_test_time += time.time() - img_start
                sum_count += 1
                if result == true_label:
                    sum_acc += 1
            break
    acc = sum_acc / sum_count if sum_count > 0 else 0
    print(f"\n文件夹 {dir_path} 测试结果：")
    print(f"  准确率: {acc:.2%} ({sum_acc}/{sum_count})")
    print(f"  总时间: {time.time() - test_start_time:.2f} 秒")
    print(f"  平均每张: {total_test_time / sum_count:.2f} 秒")
    return acc

def show_image(image_path):
    img = Image.open(image_path)
    img = transforms.Resize([224, 224])(img)
    img_np = np.array(img)
    img_np = (img_np - img_np.mean()) / img_np.std()
    plt.imshow(img_np)
    plt.show()

# ================== 核心训练函数（带日志记录） ==================
def train_model(model_builder, model_name, dataset):
    torch.cuda.empty_cache()
    print(f"\n{'='*40}")
    print(f"开始训练模型: {model_name}")
    print(f"{'='*40}")

    model = model_builder(num_class=N_CLASS).to(device)

    # 可选：torch.compile（PyTorch 2.0+）
    if hasattr(torch, 'compile'):
        print("启用 torch.compile 加速...")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 学习率调度器（余弦退火）
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    # 混合精度训练
    scaler = GradScaler()

    # ========== 新增：记录历史 ==========
    epoch_losses = []
    val_accs = []
    # ================================

    max_acc = 0.0

    for epoch in range(EPOCH):
        print(f"\n>>> 进入第 {epoch+1}/{EPOCH} 个 Epoch")
        model.train()
        epoch_loss = 0.0

        for i, (train_data, train_label, _) in enumerate(dataset.train_data_loader):
            train_data = train_data.to(device, non_blocking=True)
            train_label = train_label.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                pred = model(train_data)
                loss = loss_function(pred, train_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"  Batch [{i}/{len(dataset.train_data_loader)}] | Loss: {loss.item():.4f}")

        # 每个 epoch 结束验证一次
        print("\n  [验证环节] 正在测试当前模型性能...")
        model.eval()
        sum_acc, sum_count = 0, 0
        with torch.no_grad():
            for test_data, test_label, _ in dataset.train_test_data_loader:
                test_data = test_data.to(device, non_blocking=True)
                test_label = test_label.to(device, non_blocking=True)
                with autocast():
                    pred = model(test_data)
                pred_idx = torch.argmax(pred, dim=1)
                sum_acc += torch.sum(pred_idx == test_label).item()
                sum_count += test_label.size(0)

        acc = sum_acc / sum_count
        print(f"  [验证环节] 当前准确率: {acc:.2%}")

        # ========== 记录当前 epoch 的损失和准确率 ==========
        avg_loss = epoch_loss / len(dataset.train_data_loader)
        epoch_losses.append(avg_loss)
        val_accs.append(acc)
        # =================================================

        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"  [保存] 发现更优模型，已保存至 {model_name}_best.pth (Acc: {max_acc:.2%})")

        # 更新学习率
        scheduler.step()

        print(f"<<< 第 {epoch+1} 个 Epoch 结束，平均 Loss: {avg_loss:.4f}")

    # ========== 训练结束后保存历史数据 ==========
    import pandas as pd
    df = pd.DataFrame({
        "epoch": range(1, EPOCH + 1),
        "train_loss": epoch_losses,
        "val_acc": val_accs
    })
    csv_path = f"{model_name}_training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"训练日志已保存至 {csv_path}")
    # ============================================

    print(f"{model_name} 训练完成，最高准确率: {max_acc:.2%}")

    # 清理显存
    del model
    del optimizer
    del scaler
    torch.cuda.empty_cache()
    gc.collect()
    if hasattr(torch, '_dynamo'):
        torch._dynamo.reset()

    return max_acc

# ================== 主程序 ==================
if __name__ == '__main__':
    torch.cuda.empty_cache()

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据路径（请根据实际修改）
    train_dir_path = r'Mushroom_dataset/train'
    test_dir_path = r'Mushroom_dataset/test'

    # 实例化数据集（传入种子，确保划分一致）
    dataset = RoadTagProducer(train_dir_path, test_dir_path, BATCH_SIZE, TEST_REMAIN, seed=seed)

    # 模型列表
    models = [
        #('resnet18', resnet18),
        #('resnet34', resnet34),
        #('resnet50', resnet50),
        #('vgg19',VGG),
        #('vit_tiny', vit_tiny),
        ('swin_tiny', swin_tiny),
        ('convnext_v2_tiny', convnext_v2_tiny),
        #('resnet101', resnet101),
        #('resnet152', resnet152),
    ]

    # 可选：只训练部分模型（例如只训练 Swin）
    # models = [('swin_tiny', swin_tiny)]

    results = {}
    for name, builder in models:
        acc = train_model(builder, name, dataset)
        results[name] = acc
        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 50)
    print("所有模型训练结果汇总：")
    for name, acc in results.items():
        print(f"{name:20s}: {acc:.2%}")