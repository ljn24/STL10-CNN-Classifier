# STL10 CNN Classifier 🚀

一个基于 **PyTorch** 的 STL-10 图像分类实验，用于实践卷积神经网络建模、训练策略优化、消融实验分析与模型可解释性可视化。

项目围绕 Baseline CNN 与 Advanced CNN 展开，包含完整的训练、评估、实验配置、指标保存、训练曲线绘制和 Grad-CAM 可视化流程，便于从实验结果中分析不同模型结构与训练技巧对分类性能的影响。

## ✨ 项目亮点

- 🧠 **Baseline CNN**：使用 Conv-BN-ReLU、Pooling 和 Dropout 构建基础卷积网络。
- ⚡ **Advanced CNN**：引入残差连接、SE 通道注意力、数据增强、CutMix、AdamW、Cosine LR 和 Label Smoothing。
- 🧪 **消融实验**：通过 YAML 配置逐项移除改进模块，分析每项技术的贡献。
- 📊 **训练可视化**：自动保存训练历史，并可生成 Loss / Accuracy 曲线。
- 🔥 **Grad-CAM 可解释性**：生成类别概览、正确/错误样本对比和模型热力图对比。

## 🗂️ 项目结构

```text
.
├── configs/                 # 训练配置
│   ├── baseline.yaml
│   ├── advanced.yaml
│   └── ablation/            # 消融实验配置
├── src/
│   ├── dataset.py           # STL-10 数据加载与增强
│   ├── model.py             # Baseline / Advanced 模型
│   ├── train.py             # 训练入口
│   ├── eval.py              # 测试集评估入口
│   └── utils.py             # 指标、配置、checkpoint 工具
├── scripts/
│   ├── plot_curves.py       # 绘制训练曲线
│   └── run_all.sh           # 批量训练与评估所有配置
├── visualization/
│   ├── gradcam.py           # Grad-CAM 实现
│   ├── plotting.py          # 可视化绘图函数
│   └── visualize.py         # Grad-CAM 图像生成入口
└── outputs/                 # 训练结果、模型权重与可视化输出
```

## 🛠️ 环境准备

项目已提供 `pyproject.toml` 和 `uv.lock`，推荐直接使用 `uv sync` 同步环境：

```bash
uv sync
```

同步完成后，`uv` 会在项目中创建 `.venv`。你可以激活环境后运行命令，也可以直接使用 `uv run`：

```bash
source .venv/bin/activate
```

如果你使用 CUDA，请根据自己的显卡和 CUDA 版本确认 `pyproject.toml` 中的 PyTorch 版本是否合适。

## 📦 数据准备

代码使用 `torchvision.datasets.ImageFolder` 读取数据，默认路径为：

```text
dataset/STL10/
├── train/
│   ├── airplane/
│   ├── bird/
│   └── ...
└── test/
    ├── airplane/
    ├── bird/
    └── ...
```

如果你的数据不在默认位置，请修改对应配置文件中的 `data.root`：

```yaml
data:
  root: ./dataset/STL10
```

## 🚀 快速开始

训练 Baseline 模型：

```bash
python -m src.train --config configs/baseline.yaml
```

训练 Advanced 模型：

```bash
python -m src.train --config configs/advanced.yaml
```

训练完成后，脚本会在对应的 `outputs/` 子目录下保存：

- `best_model.pt`：验证集表现最好的模型权重
- `history.json`：每个 epoch 的训练 / 验证指标
- `config.yaml`：本次实验使用的配置副本

## ✅ 模型评估

在 STL-10 测试集上评估模型：

```bash
python -m src.eval --config configs/advanced.yaml
```

评估结果会保存到：

```text
outputs/advanced/test_metrics.json
```

其中包含测试准确率、每个类别的 precision / recall / F1-score 以及混淆矩阵。

当前已有实验结果中，Baseline 测试准确率约为 **76.2%**，Advanced 测试准确率约为 **87.5%**。

## 📈 绘制训练曲线

训练完成后，可以生成 Loss 和 Accuracy 曲线：

```bash
python -m scripts.plot_curves --config configs/advanced.yaml
```

输出文件：

```text
outputs/advanced/training_curves.png
```

## 🔥 Grad-CAM 可视化

生成单模型 Grad-CAM 报告图：

```bash
python -m visualization.visualize --config configs/advanced.yaml
```

对比 Advanced 与 Baseline 模型的关注区域：

```bash
python -m visualization.visualize \
  --config configs/advanced.yaml \
  --compare configs/baseline.yaml
```

输出目录：

```text
outputs/advanced/gradcam/
├── class_overview.png
├── correct_vs_wrong.png
└── model_comparison.png
```

## 🧪 消融实验

消融实验配置位于 `configs/ablation/`，用于分析 Advanced 模型中不同改进项的影响：

```text
configs/ablation/
├── Z1_no_residual.yaml
├── Z2_no_se.yaml
├── Z3_no_augment.yaml
├── Z4_no_cutmix.yaml
├── Z5_no_all_aug.yaml
└── Z6_no_recipe.yaml
```

运行单个消融实验：

```bash
python -m src.train --config configs/ablation/Z1_no_residual.yaml
python -m src.eval --config configs/ablation/Z1_no_residual.yaml
```

批量运行 Baseline、Advanced 和全部消融实验：

```bash
bash scripts/run_all.sh
```

## ⚙️ 配置说明

项目采用 YAML 配置驱动，常用字段如下：

```yaml
seed: 42
device: cuda:0
model_type: advanced

model:
  use_residual: true
  use_se: true

data:
  root: ./dataset/STL10
  batch_size: 64
  augment: true

train:
  epochs: 150
  lr: 0.001
  optimizer: adamw
  scheduler: cosine
  label_smoothing: 0.05
  cutmix_alpha: 0.5

output_dir: ./outputs/advanced
```

如果没有可用 GPU，可以将 `device` 改为：

```yaml
device: cpu
```

