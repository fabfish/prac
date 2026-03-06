# PRAC: Multi-Model Fine-tuning with Activation Compression

**项目定位**: 支持多种预训练模型（RoBERTa、LLaMA等）的高效微调框架，集成 PRAC 激活压缩技术实现内存优化

---

## 📋 项目概述

本项目提供一套统一的微调框架，支持在多种预训练模型上应用 **PRAC (Principal-Random Subspace)** 激活压缩技术。通过主子空间与随机子空间的双空间分解，在保持模型性能的同时实现高达 **36% 的内存减少**。

### 支持的模型

| 模型系列 | 规模 | 支持的任务类型 | 状态 |
|---------|------|--------------|------|
| **RoBERTa** | base, large | GLUE, SQuAD, 分类任务 | ✅ 完整支持 |
| **LLaMA** | 1B, 7B, 13B, 70B | 指令微调、对话、生成任务 | ✅ 完整支持 |
| **LLaMA-2** | 7B, 13B, 70B | 指令微调、对话、生成任务 | ✅ 完整支持 |
| **LLaMA-3** | 8B, 70B | 指令微调、对话、生成任务 | ✅ 完整支持 |
| **Qwen** | 1.8B - 72B | 指令微调、对话 | 🚧 实验支持 |
| **Baichuan** | 7B, 13B | 指令微调 | 🚧 实验支持 |

### 核心特性

1. **多模型统一接口**: 一套代码适配 RoBERTa 和 LLaMA 系列
2. **激活压缩**: PRAC 技术减少训练内存占用 30-40%
3. **LoRA 兼容**: 可与 LoRA/QLoRA 结合使用
4. **多任务支持**: 分类、问答、生成、指令微调
5. **混合精度**: 支持 FP16/BF16/INT8 训练

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/fabfish/prac.git
cd prac
pip install -r requirements.txt
```

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.35
- CUDA >= 11.8 (推荐)

---

## 🎯 微调指南

### 1. RoBERTa 在 GLUE 任务上微调

```bash
# SST-2 情感分类
python scripts/finetune_glue.py \
    --model_name roberta-base \
    --task sst2 \
    --use_prac \
    --principal_rank 0.3 \
    --random_rank 0.3 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --output_dir ./output/roberta-sst2

# 所有 GLUE 任务
for task in sst2 mrpc qnli rte; do
    python scripts/finetune_glue.py \
        --model_name roberta-base \
        --task $task \
        --use_prac \
        --batch_size 32 \
        --output_dir ./output/roberta-$task
done
```

### 2. RoBERTa 在 SQuAD 上微调

```bash
# SQuAD v1.1
python scripts/finetune_squad.py \
    --model_name roberta-base \
    --dataset squad \
    --use_prac \
    --principal_rank 0.3 \
    --random_rank 0.3 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 2 \
    --output_dir ./output/roberta-squad

# SQuAD v2.0
python scripts/finetune_squad.py \
    --model_name roberta-base \
    --dataset squad_v2 \
    --use_prac \
    --batch_size 16 \
    --output_dir ./output/roberta-squad-v2
```

### 3. LLaMA 指令微调

```bash
# LLaMA-2-7B + LoRA + PRAC
python scripts/finetune_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset alpaca \
    --use_prac \
    --principal_rank 0.25 \
    --random_rank 0.25 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --output_dir ./output/llama2-7b-alpaca

# LLaMA-3-8B 全参数微调（需多卡）
python scripts/finetune_llama.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --dataset dolly \
    --use_prac \
    --principal_rank 0.3 \
    --random_rank 0.3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --bf16 \
    --deepspeed configs/ds_config_zero2.json \
    --output_dir ./output/llama3-8b-dolly
```

### 4. LLaMA 对话微调

```bash
# 使用 ShareGPT 对话数据
python scripts/finetune_llama.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset sharegpt \
    --use_prac \
    --principal_rank 0.25 \
    --random_rank 0.25 \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --batch_size 2 \
    --max_seq_length 2048 \
    --output_dir ./output/llama2-7b-chat
```

---

## 🔬 PRAC 配置详解

### 压缩率选择

| 模型规模 | principal_rank | random_rank | 内存节省 | 性能影响 |
|---------|---------------|-------------|---------|---------|
| 小模型 (<1B) | 0.4 | 0.3 | ~25% | <1% |
| 中模型 (1B-7B) | 0.3 | 0.3 | ~36% | <2% |
| 大模型 (7B+) | 0.25 | 0.25 | ~40% | <3% |

### 动态配置示例

```python
from src.prac import PRACConfig

# 针对不同层设置不同压缩率
config = PRACConfig(
    principal_rank=0.3,
    random_rank=0.3,
    principal_update_freq=200,  # 每200步更新主子空间
    random_update_freq=100,     # 每100步更新随机子空间
    layer_configs={
        "attention": (0.3, 0.3),  # Attention 层
        "mlp": (0.35, 0.25),      # MLP 层
    }
)
```

---

## 📊 实验结果

### RoBERTa 在 GLUE 上的表现

| 任务 | 基线 (F1/Acc) | +PRAC (F1/Acc) | 内存节省 |
|------|--------------|----------------|---------|
| SST-2 | 94.8 | 94.6 (-0.2) | **35%** |
| MRPC | 90.2 | 90.0 (-0.2) | **36%** |
| QNLI | 92.5 | 92.3 (-0.2) | **35%** |
| RTE | 85.4 | 85.1 (-0.3) | **35%** |

### LLaMA-2-7B 指令微调

| 方法 | MT-Bench | AlpacaEval | 训练内存 |
|------|---------|-----------|---------|
| 全量微调 | 6.8 | 78.2 | 56 GB |
| LoRA | 6.5 | 75.4 | 18 GB |
| LoRA + PRAC | 6.4 | 74.8 | **12 GB** |

---

## 🏗️ 项目结构

```
prac/
├── src/
│   ├── prac.py              # PRAC 核心实现
│   ├── models/              # 模型适配器
│   │   ├── roberta.py       # RoBERTa 适配
│   │   └── llama.py         # LLaMA 适配
│   └── trainers/            # 训练器
│       ├── glue_trainer.py
│       ├── squad_trainer.py
│       └── llama_trainer.py
├── scripts/                 # 微调脚本
│   ├── finetune_glue.py     # GLUE 微调
│   ├── finetune_squad.py    # SQuAD 微调
│   └── finetune_llama.py    # LLaMA 微调
├── configs/                 # 配置文件
│   ├── prac/                # PRAC 配置
│   │   ├── roberta_base.yaml
│   │   └── llama_7b.yaml
│   └── deepspeed/           # DeepSpeed 配置
│       ├── ds_config_zero2.json
│       └── ds_config_zero3.json
├── data/                    # 数据处理
│   ├── glue_utils.py
│   ├── squad_utils.py
│   └── instruction_utils.py
├── tests/                   # 单元测试
│   └── test_prac.py
├── requirements.txt
└── README.md
```

---

## 🔧 高级用法

### 与 DeepSpeed 结合

```bash
# ZeRO-2 配置
python scripts/finetune_llama.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --use_prac \
    --use_lora \
    --deepspeed configs/deepspeed/ds_config_zero2.json
```

### 自定义数据集

```python
# data/custom_dataset.py
from datasets import load_dataset

def load_custom_dataset(data_path):
    dataset = load_dataset('json', data_files=data_path)
    return dataset

# 训练时使用
python scripts/finetune_llama.py \
    --dataset custom \
    --data_path /path/to/your/data.json \
    --use_prac
```

### 推理加速

```python
from src.prac import load_prac_model

# 加载训练好的 PRAC 模型
model = load_prac_model("./output/llama2-7b-alpaca")

# 推理（PRAC 自动禁用，不影响速度）
output = model.generate(input_ids, max_length=512)
```

---

## 📈 监控与调试

### 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir ./output

# 监控指标
# - 训练损失
# - 验证准确率
# - PRAC 内存节省比例
# - GPU 内存使用
```

### 内存分析

```python
from src.prac import analyze_memory

# 分析模型内存使用
stats = analyze_memory(model)
print(f"原始内存: {stats['original_mb']:.1f} MB")
print(f"PRAC 内存: {stats['compressed_mb']:.1f} MB")
print(f"节省比例: {stats['savings_ratio']:.1f}%")
```

---

## 🧪 测试

```bash
# 运行单元测试
pytest tests/

# 测试 PRAC 压缩效果
python tests/test_prac.py

# 基准测试
python scripts/benchmark.py --model roberta-base
```

---

## 📚 引用

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

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**