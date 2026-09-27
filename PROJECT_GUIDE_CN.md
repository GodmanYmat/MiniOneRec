# MiniOneRec 中文源码导读

> 本文面向第一次接触生成式推荐、语义 ID（Semantic ID）和 GRPO 的读者。目标不是逐行翻译代码，而是解释每个文件在系统中的职责、输入输出以及它与其他文件的关系。

## 1. 先用一句话理解项目

MiniOneRec 先把每个商品的标题和描述编码成向量，再把向量量化成类似 `<a_223><b_80><c_216>` 的三层语义 ID（简称 SID），最后训练大语言模型根据用户历史 SID 生成下一个商品 SID。

它的主流程是：

```text
Amazon 原始评论和商品信息
        │
        ▼
数据清洗、K-core 过滤、按时间构造用户序列
        │
        ▼
商品标题与描述 ──文本编码器──> 商品稠密向量
        │
        ▼
RQ-VAE / RQ-KMeans ──> 三层 Semantic ID
        │
        ▼
训练数据格式转换
        │
        ├──> SFT：学习推荐任务以及 SID 与自然语言的对应关系
        │
        └──> GRPO：根据推荐奖励继续优化模型
                         │
                         ▼
                  受约束 Beam Search
                         │
                         ▼
                    HR@K / NDCG@K
```

这不是一个 Web 服务项目，也不是传统的“召回模型 + 排序模型”工程。它更接近一套论文实验代码：以 Python 脚本为入口，通过 Shell 脚本组合多 GPU 训练流程。

---

## 2. 阅读代码前需要知道的概念

### 2.1 Item ID 与 Semantic ID

普通商品 ID 只是一个没有语义的编号：

```text
item_id = 3681
```

Semantic ID 是根据商品文本向量生成的分层编码：

```text
item_sid = <a_223><b_80><c_216>
```

这里实际上包含三个新增到 tokenizer 中的 token：

```text
<a_223>
<b_80>
<c_216>
```

第一层通常表示较粗的语义类别，后面的层对前一层尚未表达的残差信息继续细分。

### 2.2 SFT

SFT 是监督微调。模型拿到一个 Prompt 和标准答案，只在答案 token 上计算交叉熵损失。

例如：

```text
输入：用户依次交互过 SID-A、SID-B，请预测下一个商品。
答案：SID-C
```

### 2.3 GRPO

GRPO 会针对同一个 Prompt 生成多个候选，通过奖励比较候选的相对好坏：

```text
同一 Prompt → 16 个候选 → 16 个 reward → 组内标准化 advantage
```

正 advantage 会提高对应生成结果的概率，负 advantage 会降低对应生成结果的概率。项目同时使用参考模型的 KL 惩罚，防止 RL 模型过度偏离 SFT 模型。

### 2.4 受约束生成

商品空间是一个封闭集合。模型不应该生成不存在的 SID，因此代码会根据所有合法 SID 构建前缀约束：

```text
生成 <a_223> 后，只允许选择能够组成合法商品的 <b_x>；
生成 <a_223><b_80> 后，只允许选择合法的 <c_x>。
```

---

## 3. 仓库结构总览

```text
MiniOneRec/
├── README.md                     项目说明与运行流程
├── requirements.txt              Python/CUDA 依赖
├── data.py                       SFT、RL、评估数据集定义
├── sft.py / sft.sh               标准 SFT 主流程
├── rl.py / rl.sh                 GRPO 主流程
├── minionerec_trainer.py         定制 GRPO Trainer
├── evaluate.py / evaluate.sh     多 GPU 离线生成评估
├── LogitProcessor.py             合法 SID 约束解码
├── calc.py                       HR、NDCG、无效结果统计
├── split.py / merge.py           切分与合并评估数据
├── convert_dataset.py            将预处理数据转换为训练 CSV
├── *_gpr.py                      GPR 实验分支
├── ts_rec_*.py                   TS-Rec 实验分支
├── sasrec.py                     可选协同过滤奖励模型
├── data/                         数据处理代码与示例数据
├── rq/                           商品向量和 SID 构造子系统
├── config/                       Accelerate/DeepSpeed 配置
├── tests/                        少量自动化测试
└── assets/                       README 图片资源
```

---

## 4. 数据文件之间是什么关系

以 `Industrial_and_Scientific` 为例。

### 4.1 `*.item.json`

路径示例：

```text
data/Amazon/index/Industrial_and_Scientific.item.json
```

保存商品原始特征：

```json
{
  "0": {
    "title": "SUPCO SPP6 Relay/Capacitor Hard Start Kit...",
    "description": "...",
    "brand": "Sealed Unit Parts Co., Inc.",
    "categories": ""
  }
}
```

主要消费者：

- `rq/text2emb/amazon_text2emb.py`：生成商品文本向量。
- `data.py`：构造标题与 SID 的对齐任务。
- `convert_dataset.py`：把标题写入最终训练 CSV。

### 4.2 `*.emb-qwen-td.npy`

路径示例：

```text
data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy
```

这是商品文本向量矩阵。当前 Industrial 数据形状为：

```text
(3686, 2560)
```

第 `i` 行必须对应商品 `i`。这个顺序关系非常重要，后续 SID 生成代码通常默认数组行号就是商品 ID。

### 4.3 `*.index.json`

路径示例：

```text
data/Amazon/index/Industrial_and_Scientific.index.json
```

保存商品 ID 到 SID token 列表的映射：

```json
{
  "0": ["<a_236>", "<b_231>", "<c_226>"],
  "1": ["<a_42>", "<b_80>", "<c_160>"]
}
```

### 4.4 `train/valid/test/*.csv`

最终训练数据包含：

| 字段 | 含义 |
|---|---|
| `user_id` | 用户 ID |
| `history_item_title` | 历史商品标题列表 |
| `item_title` | 目标商品标题 |
| `history_item_id` | 历史商品原始 ID |
| `item_id` | 目标商品原始 ID |
| `history_item_sid` | 历史商品 SID 列表 |
| `item_sid` | 目标商品 SID |

同一行既保存原始 ID、自然语言标题和 SID，是因为项目会基于它构造多种训练任务。

### 4.5 `info/*.txt`

每行格式为：

```text
SID<TAB>商品标题<TAB>原始商品ID
```

它主要用于：

- 构建合法 SID 前缀约束。
- 把生成结果映射回商品。
- 计算离线推荐指标。

---

## 5. 根目录核心文件

### 5.1 `README.md`

项目总说明，包含：

- 方法简介。
- 数据处理到评估的完整命令。
- 支持的 SID 构造方法。
- SFT、RL、评估入口。
- GPR、TS-Rec 等新增功能说明。

初学者应该先看 README 理解流程，但真正运行前还需要检查 Shell 脚本，因为部分路径仍是占位符。

### 5.2 `requirements.txt`

记录训练环境依赖，核心库包括：

- PyTorch
- Transformers
- TRL
- Accelerate
- DeepSpeed
- Datasets
- bitsandbytes
- NumPy、Pandas、Scikit-learn

依赖明显偏向 Linux + NVIDIA GPU，并同时出现 CUDA 11 和 CUDA 12 相关包。实际部署时最好重新建立经过验证的 Conda/Docker 环境，而不是在任意机器上直接全量安装。

### 5.3 `data.py`

这是项目的数据中心，定义了 SFT、RL 和评估阶段的大部分 Dataset。文件较长，可以按功能分组理解。

#### 基础类

- `Tokenizer`：统一处理 BOS/EOS token，避免不同 tokenizer 重复添加特殊 token。
- `BaseDataset`：定义 `get_inputs()`、`__getitem__()` 等公共逻辑。
- `CSVBaseDataset`：读取最终训练 CSV。
- `JSONBaseDataset`：读取商品特征和 SID 映射。

#### 标准 SFT 数据集

- `SidSFTDataset`：输入历史 SID，输出下一个商品 SID。这是核心推荐任务。
- `SidItemFeatDataset`：构造 `SID → 标题` 和 `标题 → SID`，让 SID 与自然语言对齐。
- `FusionSeqRecDataset`：输入 SID 历史，输出下一个商品标题，连接推荐空间和语言空间。
- `TitleHistory2SidSFTDataset`：输入历史商品标题，输出目标 SID，目前标准 `sft.py` 中没有启用。
- `PreferenceSFTDataset`、`UserPreference2sidSFTDataset`：与用户偏好文本有关的辅助任务，目前标准主线未启用。

#### RL 数据集

- `SidDataset`：历史 SID 到目标 SID，返回 `prompt/completion`，不提前 tokenize。
- `RLTitle2SidDataset`：标题或描述到 SID。
- `RLSeqTitle2SidDataset`：标题序列到下一个 SID。
- `RLSid2TitleDataset`：SID 到标题，目前标准 RL 主线中被注释。
- `RLSidhis2TitleDataset`：SID 历史到标题，目前标准 RL 主线中被注释。

#### 评估数据集

- `EvalSidDataset`：生成只有 Prompt、没有目标答案的 token 输入，同时保留目标 SID 用于后续指标计算。
- `EvalD3Dataset`：以商品标题为输出空间的旧评估路径，当前默认没有启用。

#### 旧代码和实验代码

- `SFTData`、`D3Dataset`：基于自然语言商品标题的较早版本任务。
- `SidSFTDataset_GPR`：加入用户、场景和商品异构 token 的 GPR 数据集。

注意：代码多处使用 `eval()` 解析 CSV 中的列表字符串。对于仓库自产的可信数据可以工作，但更安全的实现应使用 `ast.literal_eval()`。

### 5.4 `sft.py`

标准 SFT 入口，主要执行以下步骤：

1. 设置随机种子和分布式参数。
2. 从 `base_model` 加载因果语言模型和 tokenizer。
3. 从 `*.index.json` 收集 `<a_x>/<b_x>/<c_x>` token。
4. 调用 `tokenizer.add_tokens()` 扩展词表。
5. 调用 `model.resize_token_embeddings()` 扩展 embedding 和输出层。
6. 组合三类训练数据：
   - `SidSFTDataset`
   - `SidItemFeatDataset`
   - `FusionSeqRecDataset`
7. 使用 Hugging Face `Trainer` 做标准交叉熵训练。
8. 保存模型和 tokenizer。

`freeze_LLM=True` 时，代码会冻结模型，并通过梯度 hook 只保留新增 SID embedding 行的梯度。但这种冻结状态不会自然保存在普通 `state_dict` 中；RL 重新加载模型后默认仍会恢复为全参数可训练。

### 5.5 `sft.sh`

标准 SFT 启动脚本：

- 使用 `torchrun --nproc_per_node 8` 启动 8 卡训练。
- 查找训练集、验证集、测试集和 info 文件。
- 传入 SID index 与商品元数据路径。

运行前必须替换：

```text
your_model_path
output_dir/xxx
wandb_proj
wandb_name
```

### 5.6 `rl.py`

标准 RL 入口。主要工作是：

1. 构造 RL 数据集。
2. 建立 `prompt → history → target` 映射。
3. 定义推荐奖励函数。
4. 配置 `GRPOConfig`。
5. 创建 `ReReTrainer`。
6. 调用 `trainer.train()`。

默认训练数据混合：

- SID 历史到目标 SID。
- 商品标题/描述到 SID。
- 标题序列到目标 SID。

支持的奖励：

- `rule`：精确命中 SID 得 1，否则得 0。
- `ranking`：精确命中奖励 + 排名相关负奖励。
- `ranking_only`：只使用排名奖励。
- `semantic`：预测商品和目标商品的语义向量相似度。
- `sasrec`：使用 SASRec 协同过滤分数。

代码在进入 Trainer 前还额外加载了一份 `llm_model`。在默认 ranking 奖励下它基本只被用来取得 device，可能造成额外显存占用。

### 5.7 `rl.sh`

RL 多 GPU 启动脚本，使用：

```text
Accelerate + DeepSpeed ZeRO-2
```

默认关键参数：

```text
num_generations = 16
reward_type = ranking
beam_search = True
beta = 1e-3
learning_rate = 1e-5
```

`model_path` 应该指向完成 SFT 的 checkpoint，而不是原始基础模型。

### 5.8 `minionerec_trainer.py`

这是项目最复杂的文件，继承 Hugging Face `Trainer`，实现推荐版 GRPO。

可以分成五部分理解。

#### A. Policy 与 Reference Model

- Policy model：参与梯度更新。
- Reference model：冻结，用于计算 KL 惩罚。
- 如果启用 PEFT，则可以通过暂时禁用 adapter 得到参考策略；当前标准 `rl.py` 没有启用 PEFT。

#### B. RepeatRandomSampler

将每个 Prompt 连续重复 `num_generations` 次，使同一组候选在 batch 中相邻，便于按组计算均值、标准差和 advantage。

#### C. 合法 SID 约束

读取 `info_file` 中所有 SID，构建“当前生成前缀 → 下一步合法 token”的字典。生成时通过 `ConstrainedLogitsProcessor` 把非法 token 的分数设为负无穷。

#### D. `_prepare_inputs()`

这个函数实际上承担了 GRPO 的采样与奖励准备：

1. tokenize Prompt。
2. 生成多个候选 SID。
3. 在 EOS 后构造 completion mask。
4. 计算参考模型 log probability。
5. decode 候选文本。
6. 调用 reward 函数。
7. 对同组 reward 标准化为 advantage。
8. 返回计算 loss 所需的张量。

#### E. `compute_loss()`

重新使用 Policy model 计算生成 token 的 log probability，然后计算：

```text
GRPO 策略梯度项 + beta × KL 惩罚
```

正 advantage 提高对应 SID 的生成概率，负 advantage 降低对应 SID 的生成概率。返回 loss 后，Hugging Face Trainer 负责 backward、梯度累积、梯度裁剪、optimizer step 和学习率调度。

默认情况下，RL 更新 Policy LLM 的全部参数，包括：

- SID token embedding。
- 原始 token embedding。
- Attention 参数。
- MLP/FFN 参数。
- Normalization 参数。
- LM Head。

Reference model、规则奖励、SASRec 奖励模型和固定商品向量不通过反向传播更新。

### 5.9 `LogitProcessor.py`

定义 `ConstrainedLogitsProcessor`。

每个生成步会：

1. 取出当前 beam 已经生成的 SID 前缀。
2. 调用 `prefix_allowed_tokens_fn()` 查询下一步合法 token。
3. 创建全为负无穷的 mask。
4. 仅把合法 token 的 mask 改为 0。
5. 将 mask 加到模型 logits 上。

若某个前缀找不到任何合法 token，代码会警告并尝试强制生成 EOS。

实现依赖固定的 Prompt token 前缀长度 `prefix_index=3/4`，因此更换 tokenizer 或 Prompt 格式时需要特别检查。

### 5.10 `evaluate.py`

单进程/单 GPU 评估入口：

1. 加载训练后的模型。
2. 从 info 文件读取所有合法 SID。
3. 构造 SID 前缀约束字典。
4. 使用 `EvalSidDataset` 创建测试 Prompt。
5. 使用受约束 Beam Search 生成 Top-K SID。
6. 将预测写入 JSON。

评估阶段明确使用 `do_sample=False`，目的是获得确定性的 beam 排序。

### 5.11 `evaluate.sh`

多 GPU 离线评估编排脚本：

1. `split.py` 将测试 CSV 切成多份。
2. 每张 GPU 启动一个 `evaluate.py`。
3. 等待全部 GPU 完成。
4. `merge.py` 合并结果 JSON。
5. `calc.py` 计算指标。

当前脚本按 8 张 GPU 编写，单卡环境需要修改 GPU 列表和切分数量。

### 5.12 `split.py`

读取测试 CSV，根据 GPU 列表平均切分，将子集写入：

```text
temp_dir/0.csv
temp_dir/1.csv
...
```

### 5.13 `merge.py`

读取不同 GPU 产生的 JSON 文件，按顺序合并成一个最终预测文件。

### 5.14 `calc.py`

计算推荐指标：

- HR@1、3、5、10、20、50。
- NDCG@1、3、5、10、20、50。
- CC：不在合法商品 SID 集合中的生成结果数量。

若 CC 非零，通常意味着约束生成没有正确生效，需要检查 tokenizer、Transformers 版本、Prompt 前缀长度和 `do_sample` 设置。

### 5.15 `convert_dataset.py`

连接“SID 构造阶段”和“SFT/RL 阶段”的格式转换器。

输入：

```text
Dataset.item.json
Dataset.index.json
Dataset.train.inter
Dataset.valid.inter
Dataset.test.inter
```

输出：

```text
train/*.csv
valid/*.csv
test/*.csv
info/*.txt
```

它会把原始商品 ID 序列同时转换成 SID 序列和商品标题序列。

### 5.16 `convert_dataset.sh`

`convert_dataset.py` 的命令模板。数据路径和数据集名称目前写死为 Industrial 示例，使用其他数据集时需要修改。

### 5.17 `convert_dataset_gpr.py`

GPR 版本的数据转换器。除标准字段外，还会加入：

- 原始用户标识。
- 场景 token，例如 `[CTX_HOMEPAGE]`。
- 用户、环境、商品等异构信息。

它不是标准 MiniOneRec 主线的必需步骤。

### 5.18 `sft_gpr.py`

GPR 风格的 SFT：

- 使用 `SidSFTDataset_GPR` 构造异构输入。
- 定义 `VAFT_Trainer`，对不同样本施加 value-aware 权重。
- 目标是让高价值行为对训练产生更强影响。

### 5.19 `rl_gpr.py`

GPR 风格的 RL：

- 保留标准 GRPO 奖励。
- 在 SASRec 奖励基础上加入 HEPO 层次增强奖励。
- 对用户、环境、商品层次信息进行额外建模。

建议先理解标准 `rl.py`，再阅读这个文件。

### 5.20 `ts_rec_sft.py`

TS-Rec 分支的 SFT 入口，相比标准 SFT 增加两个关键步骤。

#### Semantic-Aware Initialization

读取 SID token 对应的关键词，用关键词原有 embedding 的平均值初始化新 SID token，而不是完全随机初始化。

#### Token-Semantic Alignment

增加 `SidTokenFeatDataset`，显式训练 SID 层级 token 与关键词语义之间的对应关系。

### 5.21 `ts_rec_data.py`

TS-Rec 专用数据集实现，包含：

- `SidSFTDataset`：标准下一 SID 预测。
- `SidTokenFeatDataset`：单层 SID token 与关键词对齐。
- `SidItemFeatDataset`：商品级 SID 与文本特征对齐。
- `FusionSeqRecDataset`：行为序列与自然语言融合。

它与根目录 `data.py` 有较多重复，属于独立实验分支。

### 5.22 `ts_rec_sft.sh`

TS-Rec 启动脚本。不过当前脚本实际调用的是 `sft.py`，不是 `ts_rec_sft.py`。如果要启用 TS-Rec 的语义初始化和额外对齐任务，应优先检查并修正这个入口。

### 5.23 `sasrec.py`

实现 SASRec 及其训练/评估逻辑。MiniOneRec 主线中，它主要作为可选的协同过滤 reward model：

```text
用户历史 → SASRec → 候选商品得分 → GRPO reward
```

它不是最终执行生成推荐的模型。

### 5.24 `SASRecModules_ori.py`

SASRec 使用的底层模块，例如多头自注意力、前馈网络等。文件名中的 `ori` 表明它更像保留的原始/旧版实现。

### 5.25 `utility.py`

为 SASRec 等传统推荐代码提供辅助函数，例如：

- 序列 padding。
- 命中率计算。
- 神经过程编码器。
- Memory Unit。

其中存在一部分已经注释或未进入标准主流程的旧代码。

### 5.26 `data_test.py`

针对 `data.py` 中 Dataset 的测试和临时验证代码。它位于项目根目录，文件名也不是常见的 `test_*.py`，因此不一定会被默认的 pytest 测试发现规则自动执行。

### 5.27 `LICENSE`

项目采用 Apache 2.0 License。

---

## 6. `data/` 目录

### 6.1 `data/amazon18_data_process.py`

Amazon 2018 数据预处理主程序，也是理解原始数据如何变成推荐序列的最佳入口。

主要步骤：

1. `load_metadata_json2csv_style()`：读取商品元数据。
2. `load_reviews_json2csv_style()`：读取用户评论/交互。
3. `k_core_filtering_json2csv_style()`：执行用户和商品 K-core 过滤。
4. `convert_inters2dict_amazon18_style()`：建立用户与交互映射。
5. `generate_interaction_list_json2csv_style()`：按时间构建用户序列。
6. `convert_to_atomic_files_json2csv_style()`：划分 train/valid/test。
7. `create_item_features_amazon18_style()`：构建商品特征 JSON。

K-core 的含义是：反复删除交互数少于 K 的用户或商品，直到剩余用户和商品都满足最小交互数要求。

数据划分通常按时间进行，而不是随机划分，以模拟“用过去预测未来”。

### 6.2 `data/amazon18_data_process.sh`

Amazon 2018 预处理的参数模板，包括：

- 数据集类别。
- 用户/商品 K-core 阈值。
- 起止年月。
- 输出目录。

### 6.3 `data/amazon23_data_process.py`

Amazon Reviews 2023 格式的预处理实现。逻辑与 Amazon 2018 类似，但适配了：

- 不同的评论 JSON 格式。
- 毫秒时间戳。
- 不同的商品元数据字段。

### 6.4 `data/amazon23_data_process.sh`

Amazon 2023 数据处理启动模板，需要提供 metadata 和 reviews 文件路径。

### 6.5 `data/amazon18_data_process_gpr.py`

Amazon 2018 的 GPR 数据预处理版本，额外提取用户、环境和商品异构特征。

### 6.6 `data/process.py`

较早或辅助的数据处理脚本，包含 CSV、时间、日志等通用处理逻辑。标准 README 流程主要使用 `amazon18_data_process.py` 或 `amazon23_data_process.py`。

### 6.7 `data/Amazon/`

仓库自带的可运行示例数据：

- `index/`：商品特征、商品向量和 SID 映射。
- `train/`：训练 CSV。
- `valid/`：验证 CSV。
- `test/`：测试 CSV。
- `info/`：SID、标题和商品 ID 映射。

它允许跳过原始 Amazon 数据下载和 SID 训练，直接从 SFT 阶段开始实验。

---

## 7. `rq/`：SID 构造子系统

这里要区分“训练入口”和“模型定义”：

```text
rq/rqvae.py          训练程序
rq/models/rqvae.py   RQVAE 模型类
```

### 7.1 `rq/text2emb/amazon_text2emb.py`

商品文本向量生成程序：

1. 读取 `*.item.json`。
2. 拼接 title 和 description。
3. 用 Qwen 等模型编码文本。
4. 对非 padding token 的 hidden states 做平均池化。
5. 多进程聚合结果。
6. 按商品 ID 排序后保存 `.npy`。

这里“按商品 ID 排序”非常关键，它保证向量矩阵行号和商品 ID 一致。

### 7.2 `rq/text2emb/amazon_text2emb.sh`

使用 Accelerate 启动多 GPU 文本向量生成。模型路径仍需用户填写。

### 7.3 `rq/text2emb/amazon_text2emb_gpr.py`

GPR 风格的文本向量构造，加入异构或增强文本特征。

### 7.4 `rq/text2emb/utils.py`

文本增强和外部 LLM Provider 的辅助函数，包含：

- OpenAI。
- DeepSeek。
- MiniMax。
- 并发请求、重试、文本清洗、结果保存。

它主要用于文本特征增强，不参与标准 RQ-VAE 的前向传播。

### 7.5 `rq/datasets.py`

定义 `EmbDataset`，负责加载 `.npy` 商品向量，并以 PyTorch Dataset 形式返回每个商品向量。

### 7.6 `rq/rqvae.py`

RQ-VAE 训练入口：

1. 解析训练参数。
2. 加载 `EmbDataset`。
3. 创建 `models.rqvae.RQVAE`。
4. 创建 `rq/trainer.py` 中的 Trainer。
5. 训练并保存低重建损失或低 collision 的 checkpoint。

### 7.7 `rq/rqvae.sh`

RQ-VAE 训练命令模板。脚本使用相对路径，调用位置不正确时可能找不到 `rqvae.py` 或数据文件，运行前应确认当前工作目录。

### 7.8 `rq/trainer.py`

RQ-VAE 和 RQ-KMeans+ 共用的训练器，负责：

- optimizer 和 scheduler。
- epoch/batch 训练循环。
- 重建损失与量化损失。
- collision rate 统计。
- checkpoint 排序、保留和删除。

### 7.9 `rq/models/rqvae.py`

RQ-VAE 主模型：

```text
输入商品向量
  → MLP Encoder
  → ResidualVectorQuantizer
  → MLP Decoder
  → 重建商品向量
```

总损失为：

```text
重建损失 + quant_loss_weight × 量化损失
```

### 7.10 `rq/models/rq.py`

定义 `ResidualVectorQuantizer`。

它依次使用多层 `VectorQuantizer`：

```text
residual_0 = 编码向量
code_1 = quantizer_1(residual_0)
residual_1 = residual_0 - code_1
code_2 = quantizer_2(residual_1)
residual_2 = residual_1 - code_2
...
```

最终量化表示是各层 code vector 之和。

### 7.11 `rq/models/vq.py`

定义单层 `VectorQuantizer`：

- 计算输入向量和所有 codebook embedding 的欧氏距离。
- 普通模式选择最近的 code。
- 平衡模式通过 Sinkhorn 分配改善 code 使用不均衡。
- 计算 codebook loss 和 commitment loss。
- 使用 straight-through estimator 保留 encoder 梯度。

### 7.12 `rq/models/layers.py`

提供：

- 通用 MLP。
- 激活函数构造。
- KMeans 初始化。
- Sinkhorn 平衡算法。

### 7.13 `rq/generate_indices.py`

从训练好的 RQ-VAE checkpoint 为所有商品生成 SID：

1. 加载商品向量。
2. 调用 `model.get_indices()`。
3. 将量化索引格式化为 `<a_x>/<b_x>/<c_x>`。
4. 检查 collision。
5. 对碰撞商品尝试使用平衡分配重新编码。
6. 保存 `*.index.json`。

### 7.14 `rq/models/generate_indices.py`

与根层 `rq/generate_indices.py` 高度相似，属于重复或历史版本。标准流程应优先使用 README 指向的 `rq/generate_indices.py`。

### 7.15 `rq/rqkmeans_faiss.py`

使用 FAISS Residual Quantizer 构造 SID，特点是速度较快、无需训练神经网络，但普通语义聚类可能出现较高 collision 或聚类不平衡。

文件中还实现了：

- codebook 提取。
- residual 计算。
- Sinkhorn 均衡映射。
- SID 分布分析。

### 7.16 `rq/rqkmeans_constrained.py`

实现带容量约束的 Residual KMeans：

- 每层聚类时限制簇大小。
- 尽量让 SID 分布更均匀。
- 对重复完整 SID 增加额外 token 做去重。

### 7.17 `rq/rqkmeans_constrained.sh`

Constrained RQ-KMeans 启动脚本，支持数据集、K、层数、迭代次数和随机种子参数。

### 7.18 `rq/rqkmeans_plus.py`

RQ-KMeans+ 训练入口。它先加载 constrained RQ-KMeans 的 codebook，再训练残差编码器，使语义向量更适合已有的层次 codebook。

### 7.19 `rq/rqkmeans_plus.sh`

RQ-KMeans+ 参数模板，包含预训练 codebook 路径、三层 codebook 大小、输入维度和训练超参数。

### 7.20 `rq/generate_indices_plus.py`

加载 RQ-KMeans+ 模型生成 SID，并分析、处理重复编码。

### 7.21 `rq/generate_indices_plus.sh`

RQ-KMeans+ SID 生成模板。`ckpt_path` 当前是说明性占位符，需要替换为真实 checkpoint。

### 7.22 `rq/utils.py`

RQ 子系统的通用工具，例如：

- 创建目录。
- 彩色日志文本。
- 获取时间字符串。
- 删除旧 checkpoint。

---

## 8. 配置、测试和资源

### 8.1 `config/zero2_opt.yaml`

Accelerate + DeepSpeed 配置：

- 单机多卡。
- BF16。
- ZeRO Stage 2。
- optimizer state 和 gradient 分片。
- 不进行 CPU offload。

它主要被 `rl.sh` 使用。

### 8.2 `tests/test_minimax_provider.py`

使用 mock 测试 MiniMax Provider：

- Provider 分发。
- 请求 URL 和 payload。
- API key header。
- 重试机制。
- `<think>` 标签清理。
- temperature 限制。

它没有覆盖 SID、SFT、GRPO 或评估主流程。

### 8.3 `tests/__init__.py`

将 `tests` 标记为 Python 包，本身没有业务逻辑。

### 8.4 `assets/`

仅保存 README 使用的图片：Logo、框架图、实验结果和机构标志，不参与程序运行。

### 8.5 `ts_rec_data/Industrial_and_Scientific.description_keywords.json`

保存 TS-Rec 使用的 SID token 与关键词映射，用于：

- 根据关键词初始化 SID token embedding。
- 构造 token-semantic alignment 数据。

---

## 9. 三条最重要的调用链

### 9.1 从原始数据到 SID

```text
data/amazon18_data_process.py
    ↓
Dataset.item.json + Dataset.*.inter
    ↓
rq/text2emb/amazon_text2emb.py
    ↓
Dataset.emb-qwen-td.npy
    ↓
rq/rqvae.py
    ↓
RQ-VAE checkpoint
    ↓
rq/generate_indices.py
    ↓
Dataset.index.json
```

### 9.2 从 SID 到 SFT 模型

```text
convert_dataset.py
    ↓
train/valid/test CSV + info.txt
    ↓
sft.py
    ├── data.SidSFTDataset
    ├── data.SidItemFeatDataset
    └── data.FusionSeqRecDataset
    ↓
SFT checkpoint + 扩展后的 tokenizer
```

### 9.3 从 SFT 模型到 RL 推荐模型

```text
rl.py
    ↓
ReReTrainer
    ├── 生成多个合法 SID
    ├── 计算推荐 reward
    ├── reward 组内标准化为 advantage
    ├── 计算 Policy token log probability
    ├── 计算 Reference token log probability
    └── GRPO loss + KL loss
    ↓
Trainer.backward() + optimizer.step()
    ↓
RL checkpoint
```

---

## 10. 推荐的源码阅读顺序

### 第一阶段：先理解数据

1. 打开一行训练 CSV。
2. 查看同一商品在 `*.item.json` 中的标题和描述。
3. 查看它在 `*.index.json` 中的 SID。
4. 阅读 `convert_dataset.py`。

### 第二阶段：理解 SFT

1. 阅读 `data.py` 中的 `SidSFTDataset`。
2. 阅读 `SidItemFeatDataset`。
3. 阅读 `FusionSeqRecDataset`。
4. 阅读 `sft.py` 的 `train()`。

重点观察：

```python
labels = [-100] * input_prompt_len + answer_token_ids
```

这表示只对回答部分计算语言模型损失。

### 第三阶段：理解合法生成

1. 阅读 `evaluate.py` 如何建立 `hash_dict`。
2. 阅读 `LogitProcessor.py` 如何屏蔽非法 token。
3. 阅读 `calc.py` 如何检查 CC。

### 第四阶段：理解 GRPO

1. 先阅读 `rl.py` 中的 reward 函数。
2. 再读 `minionerec_trainer.py::_prepare_inputs()`。
3. 最后读 `minionerec_trainer.py::compute_loss()`。

不要一开始就从 `minionerec_trainer.py` 第一行顺序读到最后，这个文件混合了 TRL 兼容代码、多 GPU、vLLM、生成、奖励和评估逻辑，容易失去主线。

### 第五阶段：最后理解 SID 算法

按以下顺序阅读：

```text
rq/models/rqvae.py
    ↓
rq/models/rq.py
    ↓
rq/models/vq.py
    ↓
rq/trainer.py
    ↓
rq/generate_indices.py
```

---

## 11. 初次运行建议

如果目标是先跑通而不是从原始数据完整复现，建议使用仓库自带的 SID 和 CSV：

```text
第一步：准备兼容的 GPU 环境和基础模型
第二步：修改 sft.sh 中的模型与输出路径
第三步：运行 SFT
第四步：用 SFT final_checkpoint 修改 rl.sh 的 model_path
第五步：运行 RL
第六步：修改 evaluate.sh 的 exp_name 并评估
```

第一次实验建议：

- 先用较小基础模型。
- 单卡运行时降低 batch size，并把脚本中的 8 卡配置改为 1 卡。
- 先抽样少量训练数据验证格式和 loss。
- SFT 后先直接评估，再决定是否投入更昂贵的 RL。
- 重点检查评估日志中的 `CC` 是否为 0。

---

## 12. 阅读和修改代码时的常见陷阱

1. **商品 ID 与向量行号必须一致。** 如果商品字典顺序或重映射关系错位，生成的 SID 会绑定到错误商品。
2. **SID token 必须和 tokenizer 一起保存。** 只保存模型权重、不保存扩展 tokenizer，会导致 token ID 错位。
3. **Prompt 格式必须保持一致。** 约束生成依赖 `### Response:` 的 token 前缀。
4. **更换基础模型要重新验证前缀长度。** `prefix_index=3/4` 是模型相关假设。
5. **RL 的 batch size 必须能被 `num_generations` 整除。** Trainer 中有显式检查。
6. **全零 reward 无法提供相对策略梯度。** 如果候选从不命中，需要改善采样、SFT 初始能力或 reward 稠密度。
7. **默认 RL 是全参数训练。** 显存不足时需要考虑 LoRA、量化或减少模型副本。
8. **提供的 Shell 脚本不是即插即用配置。** 路径、GPU 数量、模型名和输出目录需要逐一确认。
9. **研究分支不要同时混用。** 标准 MiniOneRec、GPR、TS-Rec 应先分别跑通，再考虑组合。
10. **SID collision 会造成商品歧义。** 需要确认碰撞是否来自真正的重复商品，否则应使用 constrained RQ-KMeans、额外去重层或重新训练 codebook。

---

## 13. 最终应该形成的心智模型

可以把整个系统理解为三台连续工作的机器：

```text
机器一：SID 编码器
商品自然语言 ──> 结构化商品代码

机器二：生成式推荐器
用户历史商品代码 ──> 下一个商品代码

机器三：RL 排序优化器
对多个候选的相对好坏进行反馈 ──> 调整生成概率
```

项目真正的核心不只是“用 LLM 做推荐”，而是以下三个设计的组合：

1. 用分层 SID 缩小并结构化商品生成空间。
2. 用多任务 SFT 把 SID、自然语言和用户行为对齐。
3. 用合法 SID 约束和推荐 reward 对生成结果进行强化学习。

理解这三点之后，再阅读任何单个文件都会容易很多。
