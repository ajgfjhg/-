import os
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd

# ================== 配置 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "."  # 权重文件所在目录

# 模型列表（只展示带预训练权重的模型）
PRETRAINED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg19",
    "vit_tiny",
    "swin_tiny",
    "convnext_v2_tiny",
]

# 权重文件路径映射（文件名与训练时保存的一致）
MODEL_PATHS = {
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

# 类别名称（请确保与训练时的 label_map 顺序一致）
CLASS_NAMES = [
    "毒蝇鹅膏 (毒蝇伞)", "赭盖鹅膏 (食褐鹅膏)", "梨形皮马勃", "杯冠珊瑚菌",
    "美味牛肝菌", "鸡油菌", "宽鳞多孔菌", "晶粒小脆柄菇[1.1]", "毛头鬼伞 (鸡腿菇)",
    "粗糙拟迷宫菌", "金针菇 (冬菇)", "桦剥管菌 (桦孔菌)", "树舌灵芝", "大陀螺菌 (假羊肚菌)",
    "分枝猴头菇", "假鸡油菌", "毛柄库恩菌", "橙盖疣柄牛肝菌", "网状马勃",
    "高大环柄菇", "止血扇菇", "卷缘网褶菌", "平菇 (侧耳)", "晚生侧耳 (元蘑)",
    "褐乳牛肝菌 (粘盖)", "云芝 (彩绒革盖菌)", "二型齿褶菌", "红网纹拟口蘑", "皱盖钟菌"
]
NUM_CLASSES = len(CLASS_NAMES)

# ================== 模型构建函数 ==================
def build_model(model_name):
    """根据模型名称返回模型实例（未加载权重）"""
    if model_name.startswith("resnet"):
        from torchvision import models
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
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    elif model_name == "vgg19":
        from torchvision import models
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        return model

    elif model_name == "vit_tiny":
        return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES)

    elif model_name == "swin_tiny":
        return timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=NUM_CLASSES)

    elif model_name == "convnext_v2_tiny":
        return timm.create_model('convnextv2_tiny', pretrained=False, num_classes=NUM_CLASSES)

    else:
        raise ValueError(f"未知模型: {model_name}")

# ================== 模型加载与缓存 ==================
model_cache = {}

def load_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]

    model = build_model(model_name).to(DEVICE)
    weight_path = os.path.join(MODEL_DIR, MODEL_PATHS[model_name])
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    state_dict = torch.load(weight_path, map_location="cpu")
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
    model_cache[model_name] = model
    return model

# ================== 图像预处理 ==================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

# ================== 预测函数 ==================
def predict_single_model(model_name, img_tensor):
    model = load_model(model_name)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs[0], dim=0).cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    result = {}
    for idx in top3_idx:
        class_name = CLASS_NAMES[idx]
        result[class_name] = float(probs[idx])
    return result

def predict_all_models(image):
    """返回列表，每个元素为对应模型的预测结果字典（原有逻辑不变）"""
    if image is None:
        return [{"错误": "请上传图片"} for _ in PRETRAINED_MODELS]

    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    results = []
    for model_name in PRETRAINED_MODELS:
        try:
            preds = predict_single_model(model_name, img_tensor)  # 原有函数，返回top3字典
            # 提取最高置信度类别和置信度
            top_class = list(preds.keys())[0]
            top_conf = list(preds.values())[0]
            results.append([model_name, top_class, f"{top_conf:.2%}"])
        except Exception as e:
            results.append([model_name, f"错误: {e}", ""])
    # 返回 DataFrame 格式（Gradio Dataframe 要求）
    import pandas as pd
    return pd.DataFrame(results, columns=["模型名称", "识别类别", "置信度"])

# ================== 构建界面 ==================
def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="蘑菇图片识别") as demo:

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="上传蘑菇图片")
                submit_btn = gr.Button("开始识别", variant="primary")

            with gr.Column(scale=2):
                output_table = gr.Dataframe(
                    headers=["模型名称", "识别类别", "置信度"],
                    label="识别结果",
                    interactive=False,
                    wrap=True
                )

        submit_btn.click(
            fn=predict_all_models,
            inputs=[input_image],
            outputs=output_table
        )

    return demo

if __name__ == "__main__":
    print("正在预加载模型...")
    for name in PRETRAINED_MODELS:
        try:
            load_model(name)
            print(f"✓ {name} 加载成功")
        except Exception as e:
            print(f"✗ {name} 加载失败: {e}")
    print("启动界面...")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)