import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# ================== 导入你的自定义模型 ==================
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from vgg19 import VGG
from ViT import vit_tiny
from Swin import swin_tiny
from ConvNeXtV2 import convnext_v2_tiny

# ================== 配置参数 ==================
TEST_DATA_PATH = r"Mushroom_dataset/test"   # 测试集根目录
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称（必须与训练时的编号顺序一致，即文件夹名 00000~00028）
CLASS_NAMES = [
    "毒蝇鹅膏 (毒蝇伞)",          # 0
    "赭盖鹅膏 (食褐鹅膏)",        # 1
    "梨形皮马勃",                # 2
    "杯冠珊瑚菌",                # 3
    "美味牛肝菌",                # 4
    "鸡油菌",                    # 5
    "宽鳞多孔菌",                # 6
    "晶粒小脆柄菇[1.1]",        # 7
    "毛头鬼伞 (鸡腿菇)",         # 8
    "粗糙拟迷宫菌",              # 9
    "金针菇 (冬菇)",             # 10
    "桦剥管菌 (桦孔菌)",         # 11
    "树舌灵芝",                  # 12
    "大陀螺菌 (假羊肚菌)",       # 13
    "分枝猴头菇",                # 14
    "假鸡油菌",                  # 15
    "毛柄库恩菌",                # 16
    "橙盖疣柄牛肝菌",            # 17
    "网状马勃",                  # 18
    "高大环柄菇",                # 19
    "止血扇菇",                  # 20
    "卷缘网褶菌",                # 21
    "平菇 (侧耳)",               # 22
    "晚生侧耳 (元蘑)",           # 23
    "褐乳牛肝菌 (粘盖)",         # 24
    "云芝 (彩绒革盖菌)",         # 25
    "二型齿褶菌",                # 26
    "红网纹拟口蘑",              # 27
    "皱盖钟菌"                   # 28
]
NUM_CLASSES = len(CLASS_NAMES)

# 模型权重文件映射（文件名与训练时保存的完全一致）
MODEL_WEIGHTS = {
    "resnet18": "resnet18_best.pth",
    "resnet34": "resnet34_best.pth",
    "resnet50": "resnet50_best.pth",
    "resnet101": "resnet101_best.pth",
    "resnet152": "resnet152_best.pth",
    "vgg19": "vgg19_best.pth",
    "vit_tiny": "vit_tiny_best.pth",
    "swin_tiny": "swin_tiny_best.pth",
    "convnext_v2_tiny": "convnext_v2_best.pth",
}

# ================== 模型构建函数（使用你的自定义模型） ==================
def build_model(model_name):
    if model_name == "resnet18":
        return resnet18(num_class=NUM_CLASSES)
    elif model_name == "resnet34":
        return resnet34(num_class=NUM_CLASSES)
    elif model_name == "resnet50":
        return resnet50(num_class=NUM_CLASSES)
    elif model_name == "resnet101":
        return resnet101(num_class=NUM_CLASSES)
    elif model_name == "resnet152":
        return resnet152(num_class=NUM_CLASSES)
    elif model_name == "vgg19":
        return VGG(num_class=NUM_CLASSES)
    elif model_name == "vit_tiny":
        return vit_tiny(num_class=NUM_CLASSES)
    elif model_name == "swin_tiny":
        return swin_tiny(num_class=NUM_CLASSES)
    elif model_name == "convnext_v2_tiny":
        return convnext_v2_tiny(num_class=NUM_CLASSES)
    else:
        raise ValueError(f"未知模型: {model_name}")

# ================== 评估函数 ==================
def evaluate_per_class(model, loader, device, num_classes):
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.int64).to(device)
    total = torch.zeros(num_classes, dtype=torch.int64).to(device)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                if preds[i] == label:
                    correct[label] += 1

    class_acc = (correct.float() / total.float()).cpu().numpy()
    total_correct = correct.sum().item()
    total_samples = total.sum().item()
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return class_acc, overall_acc

# ================== 主程序 ==================
def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：测试集路径 {TEST_DATA_PATH} 不存在！")
        return

    # 加载测试集
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 验证类别数量
    if len(test_dataset.classes) != NUM_CLASSES:
        print(f"警告：测试集文件夹数量 {len(test_dataset.classes)} 与类别列表数量 {NUM_CLASSES} 不一致")
        print("请确保 CLASS_NAMES 顺序与文件夹编号顺序一致。")

    # 存储结果
    all_class_acc = {}
    overall_accs = {}

    # 依次评估每个模型
    for name, weight_file in MODEL_WEIGHTS.items():
        if not os.path.exists(weight_file):
            print(f"跳过：找不到权重文件 {weight_file}")
            continue

        print(f"\n---> 正在评估模型: {name}")
        try:
            model = build_model(name).to(DEVICE)
            state_dict = torch.load(weight_file, map_location=DEVICE)
            # 处理可能的键名前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('_orig_mod.'):
                    k = k[10:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            class_acc, overall_acc = evaluate_per_class(model, test_loader, DEVICE, NUM_CLASSES)
            all_class_acc[name] = class_acc
            overall_accs[name] = overall_acc

            print(f"  整体准确率: {overall_acc*100:.2f}%")
            # 打印前5个类别作为快速检查
            for i in range(min(5, NUM_CLASSES)):
                print(f"    {CLASS_NAMES[i]}: {class_acc[i]*100:.2f}%")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  评估失败: {e}")

    if not all_class_acc:
        print("没有成功评估任何模型，请检查权重文件是否存在。")
        return

    # 构建 DataFrame
    df = pd.DataFrame(all_class_acc, index=CLASS_NAMES)
    df.loc['整体准确率'] = overall_accs
    # 格式化为百分比
    df = df.applymap(lambda x: f"{x*100:.2f}%")

    print("\n" + "="*80)
    print("各类别准确率对比表 (单位: %)")
    print(df.to_string())
    print("="*80)

    # 保存文件
    csv_file = "model_per_class_accuracy.csv"
    excel_file = "model_per_class_accuracy.xlsx"
    df.to_csv(csv_file, encoding='utf-8-sig')
    try:
        df.to_excel(excel_file)
        print(f"\n表格已保存为 {csv_file} 和 {excel_file}")
    except ImportError:
        print(f"\n表格已保存为 {csv_file}（未安装 openpyxl，无法保存 Excel 文件）")

if __name__ == "__main__":
    main()