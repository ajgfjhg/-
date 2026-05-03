# 基于深度学习的蘑菇种类识别

这是一个用于毕业设计的蘑菇种类识别项目，主要基于深度学习模型完成蘑菇图像分类实验与结果评估。

## 项目内容

- 多种深度学习模型结构代码，包括 ResNet、VGG19、Swin Transformer、ViT、ConvNeXtV2 等。
- 训练、评估与结果统计脚本。
- 部分训练日志、分类准确率统计表和模型评估结果。

## 主要文件

- `train_model.py` / `train_model_a.py`：模型训练脚本。
- `evaluate_models.py` / `evaluate_models_a.py`：模型评估脚本。
- `eval_all_metrics.py`：综合指标评估脚本。
- `generate_accuracy_table.py` / `generate_accuracy_table_a.py`：准确率表格生成脚本。
- `ui.py`：识别界面相关代码。

## 说明

由于 GitHub 普通 Git 仓库存在单文件 100MB 限制，并且数据集与模型权重体积较大，本仓库未上传以下内容：

- `Mushroom_dataset/` 数据集目录
- `*.pth` 模型权重文件
- 测试输出结果和 Python 缓存文件

如需复现实验，需要自行准备数据集和训练好的模型权重，并按代码中的路径配置进行放置。

