import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================== 配置参数 ==================
TEST_DATA_PATH = r"Mushroom_dataset/test"   # 修改为你的测试集路径
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称（按训练时的顺序）
CLASS_NAMES = [
    "毒蝇鹅膏 (毒蝇伞)", "赭盖鹅膏 (食褐鹅膏)", "梨形皮马勃", "杯冠珊瑚菌", "美味牛肝菌",
    "鸡油菌", "宽鳞多孔菌", "晶粒小脆柄菇[1.1]", "毛头鬼伞 (鸡腿菇)", "粗糙拟迷宫菌",
    "金针菇 (冬菇)", "桦剥管菌 (桦孔菌)", "树舌灵芝", "大陀螺菌 (假羊肚菌)", "分枝猴头菇",
    "假鸡油菌", "毛柄库恩菌", "橙盖疣柄牛肝菌", "网状马勃", "高大环柄菇",
    "止血扇菇", "卷缘网褶菌", "平菇 (侧耳)", "晚生侧耳 (元蘑)", "褐乳牛肝菌 (粘盖)",
    "云芝 (彩绒革盖菌)", "二型齿褶菌", "红网纹拟口蘑", "皱盖钟菌"
]

# 模型权重文件映射
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

# ================== 模型构建函数 ==================
def build_model(model_name, num_classes):
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet34":
            model = models.resnet34(weights=None)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
        elif model_name == "resnet101":
            model = models.resnet101(weights=None)
        elif model_name == "resnet152":
            model = models.resnet152(weights=None)
        else:
            raise ValueError(f"未知 ResNet: {model_name}")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    elif model_name == "vit_tiny":
        return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)

    elif model_name == "swin_tiny":
        return timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

    elif model_name == "convnext_v2_tiny":
        return timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_classes)

    else:
        raise ValueError(f"未知模型: {model_name}")

# ================== 评估函数 ==================
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

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

    num_classes = len(CLASS_NAMES)
    if len(test_dataset.classes) != num_classes:
        print(f"警告：测试集文件夹数量 {len(test_dataset.classes)} 与类别列表数量 {num_classes} 不一致")
        print("请确保测试集文件夹名按编号排序，且 CLASS_NAMES 顺序与之匹配。")

    results = []

    for name, weight_file in MODEL_WEIGHTS.items():
        if not os.path.exists(weight_file):
            print(f"跳过：找不到权重文件 {weight_file}")
            continue

        print(f"\n---> 正在评估模型: {name}")
        try:
            model = build_model(name, num_classes).to(DEVICE)
            state_dict = torch.load(weight_file, map_location=DEVICE)
            # 处理键名前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('_orig_mod.'):
                    k = k[10:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()

            acc, prec, rec, f1 = evaluate_model(model, test_loader, DEVICE)
            results.append([
                name,
                f"{acc*100:.2f}%",
                f"{prec*100:.2f}%",
                f"{rec*100:.2f}%",
                f"{f1*100:.2f}%"
            ])
            print(f"  准确率: {acc*100:.2f}% | 精确率: {prec*100:.2f}% | 召回率: {rec*100:.2f}% | F1: {f1*100:.2f}%")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  评估失败: {e}")

    if not results:
        print("没有成功评估任何模型，请检查权重文件是否存在。")
        return

    # 构建 DataFrame
    df = pd.DataFrame(results, columns=["模型名称", "准确率", "精确率", "召回率", "F1分数"])
    print("\n" + "="*80)
    print("模型整体性能对比表")
    print(df.to_string(index=False))
    print("="*80)

    # 保存文件
    csv_file = "model_overall_metrics.csv"
    excel_file = "model_overall_metrics.xlsx"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    try:
        df.to_excel(excel_file, index=False)
        print(f"\n表格已保存为 {csv_file} 和 {excel_file}")
    except ImportError:
        print(f"\n表格已保存为 {csv_file}（未安装 openpyxl，无法保存 Excel 文件）")

if __name__ == "__main__":
    main()