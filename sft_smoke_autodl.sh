#!/usr/bin/env bash
# 单卡 SFT 冒烟测试。用 bash 执行；--dry-run 仅展示命令，不加载模型。
# 可覆盖：MODEL_PATH、OUTPUT_DIR、CATEGORY、CUDA_VISIBLE_DEVICES、PYTHON_BIN。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "用法：bash sft_smoke_autodl.sh [--dry-run]" >&2
  exit 2
fi

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
case "$CATEGORY" in
  Industrial_and_Scientific|Office_Products) ;;
  *) echo "此脚本仅支持仓库自带的 Industrial_and_Scientific / Office_Products" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-0.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/sft-smoke-${CATEGORY}-$(date +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

TRAIN_FILE="$PROJECT_ROOT/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="$PROJECT_ROOT/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
INDEX_FILE="$PROJECT_ROOT/data/Amazon/index/${CATEGORY}.index.json"
ITEM_FILE="$PROJECT_ROOT/data/Amazon/index/${CATEGORY}.item.json"

# 直接启动一个进程，避免读取已有的 Accelerate 多卡配置。
# 三个训练 Dataset 各抽取 128 条；batch_size=8 / micro_batch_size=1 → 累积 8 次。
command=(
  "$PYTHON_BIN" "$PROJECT_ROOT/sft.py"
  --base_model "$MODEL_PATH"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --sid_index_path "$INDEX_FILE"
  --item_meta_path "$ITEM_FILE"
  --output_dir "$OUTPUT_DIR"
  --category "$CATEGORY"
  --sample 128
  --batch_size 8
  --micro_batch_size 1
  --num_epochs 1
  --learning_rate 1e-5
  --cutoff_len 256
  --eval_step 0.5
  --train_from_scratch False
  --freeze_LLM False
  --bf16 True
  --fp16 False
  --seed 42
  --wandb_run_name sft-smoke
)

printf '执行命令：\n'
printf '%q ' "${command[@]}"
printf '\n输出目录：%s\n' "$OUTPUT_DIR"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "仅预览；未检查模型文件、依赖或 GPU，未创建输出目录。"
  exit 0
fi

if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "此脚本只用于单进程，请不要通过多卡 torchrun 启动。" >&2
  exit 1
fi

# 在模型加载前检查数据、权重、依赖和硬件。
"$PYTHON_BIN" - "$MODEL_PATH" "$TRAIN_FILE" "$EVAL_FILE" "$INDEX_FILE" "$ITEM_FILE" "$OUTPUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

model, train, valid, index, item, output = map(Path, sys.argv[1:])
for path in (model / "config.json", train, valid, index, item):
    if not path.is_file():
        raise SystemExit(f"缺少文件：{path}")
if not (list(model.glob("*.safetensors")) or list(model.glob("pytorch_model*.bin"))):
    raise SystemExit(f"未找到模型权重：{model}。请下载完整模型，而不仅是 tokenizer。")
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit(f"输出路径已被占用，请设置新的 OUTPUT_DIR：{output}")
for path in (train, valid):
    with path.open(newline="", encoding="utf-8") as f:
        count = sum(1 for _ in csv.DictReader(f))
    if count < 128:
        raise SystemExit(f"{path} 只有 {count} 条，无法抽样 128 条")

import torch
import sft  # 同时检查训练入口的实际导入依赖。
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(str(model), local_files_only=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("需要且只能暴露一张 CUDA 显卡，请检查 CUDA_VISIBLE_DEVICES")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("当前 GPU/环境不支持此脚本所需的 BF16")
print(f"预检通过：torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}")
PY

mkdir -p "$OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
echo "训练结束，请检查：$OUTPUT_DIR/final_checkpoint"
