# PRAC: Principal-Random Subspace for LLM Activation Compression

**论文实现**: PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training  
**作者**: Yanyi Li, Yimu Zhang, Cong Fang  
**发布日期**: 2026-02-26

---

## 📋 项目概述

PRAC 是一种创新的 LLM 激活值压缩方法，通过结合**主子空间**（Principal Subspace）和**随机子空间**（Random Subspace），在保持模型性能的同时实现高达 **36% 的内存减少**。

### 核心创新

1. **双空间分解**: 将激活值分解为主子空间（SVD捕获）和随机子空间（正交补空间采样）
2. **无偏估计**: 理论证明 PRAC 提供具有最小方差的无偏梯度估计
3. **惰性更新**: 动态子空间更新策略，计算开销极小
4. **子空间共享**: 跨层共享投影矩阵，进一步节省内存

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/fabfish/prac-paper.git
cd prac-paper
pip install -r requirements.txt
```

### 基本使用

```python
import torch
from src.prac import PRACConfig, PRACCompressor

# 配置 PRAC
config = PRACConfig(
    principal_rank=0.3,  # 主子空间保留 30%
    random_rank=0.3,     # 随机子空间 30%
)

# 创建压缩器
compressor = PRACCompressor(config, feature_dim=768)

# 模拟激活值
activations = torch.randn(2, 128, 768)

# 压缩
XQ1, XQ2 = compressor.compress(activations)

# 重建
reconstructed = compressor.decompress(XQ1, XQ2)
```

### 训练模型

```bash
# 使用 PRAC 训练 GPT-2
python src/train.py \
    --model_name gpt2 \
    --use_prac \
    --principal_rank 0.3 \
    --random_rank 0.3 \
    --batch_size 8 \
    --output_dir ./output

# 与基线对比 (不使用 PRAC)
python src/train.py \
    --model_name gpt2 \
    --batch_size 8 \
    --output_dir ./output_baseline
```

---

## 📊 实验结果

### 内存节省

| 模型 | 原始内存 | PRAC 内存 | 节省比例 |
|------|----------|-----------|----------|
| LLaMA-130M | 100% | 64% | **36%** |
| LLaMA-350M | 100% | 65% | **35%** |
| LLaMA-1B | 100% | 66% | **34%** |
| GPT-2-124M | 100% | 64% | **36%** |

### 性能对比

与现有方法对比（预训练困惑度）：

| 方法 | LLaMA-1B (WikiText-2) | 收敛速度 |
|------|------------------------|----------|
| 全量训练 | 28.5 | 基准 |
| GaLore | 30.2 | 慢 |
| RSO | 29.8 | 慢 |
| **PRAC** | **28.7** | **快** |

### 关键发现

- ✅ **无偏估计**: PRAC 提供理论保证的无偏梯度估计
- ✅ **最小方差**: 在激活退化条件下达到最优方差
- ✅ **计算高效**: 额外计算开销 < 2%
- ✅ **兼容性强**: 可与 LoRA、Adam-mini 等方法结合

---

## 🔬 核心算法

### 数学原理

**1. 激活值分解**

对激活值矩阵 X 进行分解：

```
X = X_principal + X_random
```

**2. 主子空间 (PAC)**

通过 SVD 提取前 r₁ 个主成分：

```
X = UΣV^T
Q₁ = V[:, :r₁]  # 主子空间投影矩阵
```

**3. 随机子空间 (RAC)**

从正交补空间随机采样 r₂ 维子空间：

```
Q₂ ~ Uniform({Q | Q^T Q = I, Q^T Q₁ = 0})
```

**4. PRAC 重建**

```
X̃ = (XQ₁)Q₁^T + k(XQ₂)Q₂^T
```

其中缩放因子 `k = (n - r₁) / r₂` 确保无偏性。

### 理论保证

**定理 1** (无偏性): 在激活退化条件下，PRAC 产生无偏梯度估计。

**定理 2** (最优性): PRAC 在所有无偏估计中具有最小方差。

---

## 🏗️ 项目结构

```
prac-paper/
├── src/
│   ├── prac.py              # 核心 PRAC 实现
│   ├── train.py             # 训练脚本
│   └── __init__.py
├── experiments/             # 实验脚本
│   ├── pretrain.py
│   ├── finetune.py
│   └── benchmark.py
├── tests/                   # 单元测试
│   └── test_prac.py
├── docs/                    # 文档
│   ├── paper_summary.md
│   └── api_reference.md
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🔧 高级配置

### 分层配置

不同层使用不同压缩率：

```python
config = PRACConfig(
    layer_configs={
        "attention": (0.3, 0.3),  # Attention 层
        "mlp": (0.4, 0.2),        # MLP 层
        "norm": (0.2, 0.2),       # LayerNorm
    }
)
```

### 与 LoRA 结合

```bash
python src/train.py \
    --model_name meta-llama/Llama-2-7b \
    --use_prac \
    --use_lora \
    --lora_r 16 \
    --principal_rank 0.3 \
    --batch_size 4
```

### 与梯度检查点结合

```python
from torch.utils.checkpoint import checkpoint

# PRAC + 梯度检查点 = 极致内存效率
model.gradient_checkpointing_enable()
```

---

## 📈 监控与调试

### 内存监控

```python
# 获取 PRAC 统计信息
stats = model.get_prac_stats()
for layer_name, stat in stats.items():
    print(f"{layer_name}: {stat['savings_ratio']:.1f}% 节省")
```

### TensorBoard

```bash
tensorboard --logdir ./output
```

可视化指标：
- 内存使用量
- 重建误差
- 训练损失
- 学习率

---

## 🧪 复现论文结果

### 预训练实验

```bash
# LLaMA-1B 预训练
python experiments/pretrain.py \
    --model_config configs/llama_1b.json \
    --use_prac \
    --dataset wikitext \
    --batch_size 32 \
    --max_steps 100000
```

### 微调实验

```bash
# GLUE 基准微调
python experiments/finetune.py \
    --model_name roberta-base \
    --task glue \
    --use_prac \
    --principal_rank 0.25
```

### 消融实验

```bash
# 对比不同配置
python experiments/benchmark.py \
    --configs configs/ablation/*.yaml
```

---

## 📚 参考文献

```bibtex
@article{li2026prac,
  title={PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training},
  author={Li, Yanyi and Zhang, Yimu and Fang, Cong},
  journal={arXiv preprint arXiv:2602.23111},
  year={2026}
}
```

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- 论文作者: Yanyi Li, Yimu Zhang, Cong Fang
- 灵感来源: GaLore, RSO, CompAct
- 开源社区

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
