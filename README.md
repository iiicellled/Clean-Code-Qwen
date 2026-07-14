# Clean Code Qwen，一个 Qwen Coder 精简代码生成 SFT + DPO 项目

本项目基于 `Qwen/Qwen2.5-Coder-7B-Instruct`，面向 Python 代码生成与代码简化任务完成了两阶段微调：

1. **SFT（监督微调）**：让模型学习给定函数需求后输出正确、简洁、可读的 Python 代码。
2. **DPO（直接偏好优化）**：在 SFT 适配器基础上继续做偏好优化，使模型更稳定地输出纯代码，并减少冗余解释、Markdown 包裹和不必要的复杂实现。

当前仓库中已经保留了 SFT 和 DPO 的训练日志、LoRA 适配器以及评估结果；同时开源了面向 Python 精简代码生成任务的 SFT 与 DPO 微调数据集，便于复现实验、扩展偏好数据和迁移到其他代码模型。

## 当前状态

- 基座模型：`models/Qwen2.5-Coder-7B-Instruct`
- SFT 配置：`configs/sft_lora.yaml`
- DPO 配置：`configs/dpo_lora.yaml`
- SFT 适配器：`output_models/qwen-coder-simplifier-lora`
- DPO 适配器：`output_models/qwen-coder-simplifier-dpo-lora`
- 训练日志：`sft.log`、`dpo.log`
- SFT 数据集：`data/python_simple_coder/sft`
- DPO 偏好数据集：`data/python_simple_coder/dpo`
- SFT 评估生成结果：`output_results/sft-evaluation`
- 最终 DPO 对比评估：`output_results/dpo-final-evaluation`

## 项目贡献

本项目不仅完成了 Qwen Coder 的 SFT 与 DPO 微调，还开源了配套的精简代码微调数据集：

- **SFT 数据集**：面向指令监督微调，样本由 `instruction` 和 `output` 组成，目标是训练模型直接输出正确、简洁的 Python 函数代码。
- **DPO 数据集**：面向偏好优化，样本由 `prompt`、`chosen` 和 `rejected` 组成，目标是让模型偏好更正确、更符合接口、更简洁、更少解释性文本的答案。
- **统一评估集**：保留 base、SFT、DPO 三个模型的生成结果和评估报告，便于复现实验结论和继续扩展数据。

数据集格式在后文“开源数据集与格式”中统一说明。

## 最终三个模型评估结果

最终评估比较了 base、SFT、DPO 三个版本在 150 条 DPO final validation 样本上的表现。评估脚本统计语法、接口、函数名、函数签名、纯代码输出、输出长度，以及额外的功能断言正确率。

| 模型 | 语法正确 | 接口存在 | 名称匹配 | 签名匹配 | 纯代码输出率 | 平均行数 | 平均字符数 | 行数差值 | 字符差值 | 功能正确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 148/150 (98.7%) | 1/150 (0.7%) | 5.08 | 173.9 | +1.31 | +35.4 | 99/150 (66.0%) |
| sft | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 147/150 (98.0%) | 150/150 (100.0%) | 4.59 | 160.0 | +0.82 | +21.4 | 101/150 (67.3%) |
| dpo | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 147/150 (98.0%) | 150/150 (100.0%) | 4.48 | 155.2 | +0.71 | +16.6 | 101/150 (67.3%) |

其中“行数差值”和“字符差值”表示模型输出相对偏好答案 `chosen` 的平均差值。DPO 的平均输出长度最接近 `chosen`，说明偏好优化进一步压缩了冗余内容。

按偏好数据类型拆分后，三类样本上的签名匹配率和纯代码输出率如下：

| 偏好类型 | base | sft | dpo |
|---|---:|---:|---:|
| correctness_protection | 签名 100.0%，纯代码 1.7% | 签名 100.0%，纯代码 100.0% | 签名 100.0%，纯代码 100.0% |
| interface_and_format_protection | 签名 87.5%，纯代码 0.0% | 签名 81.2%，纯代码 100.0% | 签名 81.2%，纯代码 100.0% |
| simplicity_preference | 签名 100.0%，纯代码 0.0% | 签名 100.0%，纯代码 100.0% | 签名 100.0%，纯代码 100.0% |

从结果看，SFT/DPO 后模型的功能正确率从 66.0% 提升到 67.3%。更明显的收益是输出格式被显著约束：base 模型几乎总会输出解释或 Markdown，而 SFT 和 DPO 都能稳定输出纯 Python 代码。DPO 相比 SFT 平均输出更短，更接近偏好数据中的简洁答案。

完整报告见：`output_results/dpo-final-evaluation/report.md`

## 目录结构

```text
sft_lora_coder/
  configs/
    sft_lora.yaml              # SFT 训练配置
    dpo_lora.yaml              # DPO 训练配置
  data/
    python_simple_coder/
      sft/
        python_simple_coder.jsonl
        python_simple_coder_train.jsonl
        python_simple_coder_valid.jsonl
        python_simple_coder_valid_tests.jsonl
      dpo/
        python_simple_coder_dpo_train.jsonl
        python_simple_coder_dpo_final_valid.jsonl
        python_simple_coder_dpo_ori.jsonl
        extra_valid_data.jsonl
  models/
    Qwen2.5-Coder-7B-Instruct/ # 本地基座模型
  output_models/
    qwen-coder-simplifier-lora/
    qwen-coder-simplifier-dpo-lora/
  output_results/
    sft-evaluation/
    dpo-final-evaluation/
  src/
    sft_train.py
    dpo_train.py
    eval_sft.py
    eval_dpo_final.py
  requirements.txt
  README.md
```

## 环境准备

推荐使用 Python 3.10 或 3.11。

```powershell
cd sft_lora_coder
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `transformers==4.44.2`
- `trl==0.8.6`
- `peft`
- `datasets`
- `accelerate`
- `bitsandbytes`

如果安装 `bitsandbytes` 遇到问题，可以保持配置中的 `load_in_4bit: false`，使用全精度/半精度加载模型。

## 开源数据集与格式

本项目开源了两类精简代码微调数据集，统一保存在 `data/python_simple_coder` 下：

- `sft/`：监督微调数据，用于让模型学习“根据函数需求直接输出正确、简洁的 Python 代码”。
- `dpo/`：偏好优化数据，用于让模型在两个候选答案中偏好更正确、更符合接口、更简洁、更少解释性文本的答案。

两类数据都采用 JSONL 格式，即每一行是一个独立 JSON 对象，方便用 `datasets`、`jsonlines` 或普通流式读取方式加载。

### SFT 数据格式

SFT 数据每行至少包含 `instruction` 和 `output`：

```json
{"instruction":"Write a function with exact signature:\ndef add(a, b)","output":"def add(a, b):\n    return a + b"}
```

字段含义：

- `instruction`：用户侧任务描述，通常包含函数功能、函数名、参数签名和必要约束。
- `output`：期望模型生成的答案，只保留 Python 代码，不包含 Markdown 或额外解释。
- `id`：可选字段，用于追踪样本来源和评估结果。

训练脚本会把 SFT 样本转成 Qwen chat template，并默认只对 assistant 的回答部分计算 loss，避免模型学习复制 prompt。

### DPO 数据格式

DPO 数据每行至少包含 `prompt`、`chosen`、`rejected`：

```json
{"prompt":"Write a function with exact signature:\ndef is_even(n)","chosen":"def is_even(n):\n    return n % 2 == 0","rejected":"def is_even(n):\n    if n % 2 == 0:\n        return True\n    else:\n        return False"}
```

字段含义：

- `prompt`：用户侧任务描述，和 SFT 的 `instruction` 作用相同。
- `chosen`：偏好的答案，要求正确、接口匹配、简洁，并尽量只输出代码。
- `rejected`：不偏好的答案，可能存在冗余、格式不稳定、接口不匹配或边界条件问题。
- `pair_type`：可选字段，用于标记偏好对类型。
- `error_type`：可选字段，用于记录 rejected 答案的主要问题。

当前 DPO 数据主要覆盖三类偏好：

- `correctness_protection`：保护正确性和边界条件。
- `interface_and_format_protection`：保护函数名、签名和纯代码格式。
- `simplicity_preference`：偏好更简洁但仍正确的实现。

这种设计让数据集既能支持 SFT 阶段的指令学习，也能支持 DPO 阶段的偏好对齐；重点不是泛化到所有代码任务，而是围绕“精简、正确、纯代码”的 Python 函数生成场景做可复现的微调实验。
## SFT 训练

当前 SFT 配置位于 `configs/sft_lora.yaml`，关键设置如下：

- 训练数据：`data/python_simple_coder/sft/python_simple_coder_train.jsonl`（约1500条数据）
- 验证数据：`data/python_simple_coder/sft/python_simple_coder_valid.jsonl`（共100条数据）
- 输出目录：`output_models/qwen-coder-simplifier-lora`
- LoRA 模块：`q_proj`、`v_proj`、`o_proj`
- LoRA rank：`8`
- 最大长度：`2048`
- 训练轮数：`1`
- best model 选择指标：`eval_loss`

运行命令：

```powershell
nohup python -m src.sft_train --config configs/sft_lora.yaml > sft.log 2>&1 & echo $! > sft_train.pid
```

训练完成后会在 `output_models/qwen-coder-simplifier-lora` 中保存 LoRA 适配器、tokenizer、训练状态和评估指标。

## DPO 训练

DPO 在 SFT 适配器基础上继续训练。当前配置位于 `configs/dpo_lora.yaml`，关键设置如下：

- 初始 SFT 适配器：`output_models/qwen-coder-simplifier-lora`
- DPO 训练数据：`data/python_simple_coder/dpo/python_simple_coder_dpo_train.jsonl`
- 最终验证数据：`data/python_simple_coder/dpo/python_simple_coder_dpo_final_valid.jsonl`
- 输出目录：`output_models/qwen-coder-simplifier-dpo-lora`
- DPO beta：`0.05`
- loss 类型：`sigmoid`
- 最大总长度：`2048`
- 最大 prompt 长度：`1536`
- 最大回答长度：`512`
- 学习率：`2.0e-6`

运行命令：

```powershell
nohup python -m src.dpo_train --config configs/dpo_lora.yaml > dpo_more.log 2>&1 & echo $! > dpo_train.pid
```

DPO 训练完毕后。模型权重相关文件保存在 `output_models/qwen-coder-simplifier-dpo-lora`。

## 评估

### SFT 评估

SFT 评估脚本会比较 base 和 SFT 模型，生成结果默认写入 `output_results/sft-evaluation`。

```powershell
python -m src.eval_sft `
  --base-model models/Qwen2.5-Coder-7B-Instruct `
  --adapter output_models/qwen-coder-simplifier-lora `
  --output-dir output_results/sft-evaluation
```

脚本支持缓存生成结果；如需重新生成，可加 `--overwrite`。

### DPO 最终评估

最终评估脚本会比较 base、SFT、DPO 三个版本，默认输出到 `output_results/dpo-final-evaluation`。

```powershell
python -m src.eval_dpo_final `
  --base-model models/Qwen2.5-Coder-7B-Instruct `
  --sft-adapter output_models/qwen-coder-simplifier-lora `
  --dpo-adapter output_models/qwen-coder-simplifier-dpo-lora `
  --tasks data/python_simple_coder/dpo/python_simple_coder_dpo_final_valid.jsonl `
  --output-dir output_results/dpo-final-evaluation
```

输出文件包括：

- `base_generations.jsonl`
- `sft_generations.jsonl`
- `dpo_generations.jsonl`
- `report.json`
- `report.md`

