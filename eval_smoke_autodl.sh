#!/usr/bin/env bash
# MODEL_PATH 必须指向 SFT/RL 的完整 final_checkpoint（含扩词后的 tokenizer）。
# 用法：MODEL_PATH=/path/to/final_checkpoint bash eval_smoke_autodl.sh [--dry-run]
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "用法：MODEL_PATH=/path/to/final_checkpoint bash eval_smoke_autodl.sh [--dry-run]" >&2
  exit 2
fi
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "请设置 MODEL_PATH，指向训练生成的 final_checkpoint 目录。" >&2
  exit 2
fi

CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
case "$CATEGORY" in
  Industrial_and_Scientific|Office_Products) ;;
  *) echo "此脚本仅支持 Industrial_and_Scientific / Office_Products" >&2; exit 2 ;;
esac
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/results/eval-smoke-${CATEGORY}-$(date +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TEST_FILE="$PROJECT_ROOT/data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="$PROJECT_ROOT/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
INDEX_FILE="$PROJECT_ROOT/data/Amazon/index/${CATEGORY}.index.json"
SUBSET_FILE="$OUTPUT_DIR/test-32.csv"
RESULT_FILE="$OUTPUT_DIR/result.json"
command=(
  "$PYTHON_BIN" "$PROJECT_ROOT/evaluate.py"
  --base_model "$MODEL_PATH"
  --info_file "$INFO_FILE"
  --category "$CATEGORY"
  --test_data_path "$SUBSET_FILE"
  --result_json_data "$RESULT_FILE"
  --batch_size 2
  --num_beams 10
  --max_new_tokens 16
  --length_penalty 0.0
  --seed 42
)
printf '执行命令：\n'
printf '%q ' "${command[@]}"
printf '\n输出目录：%s\n' "$OUTPUT_DIR"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "仅预览；未检查 checkpoint、依赖或 GPU，未创建输出目录。"
  exit 0
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "此脚本只用于单进程评估。" >&2
  exit 1
fi

"$PYTHON_BIN" - "$MODEL_PATH" "$TEST_FILE" "$INFO_FILE" "$INDEX_FILE" "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

model, test, info, index, output = map(Path, sys.argv[1:])
for path in (model / 'config.json', test, info, index):
    if not path.is_file():
        raise SystemExit(f'缺少文件：{path}')
if not (list(model.glob('*.safetensors')) or list(model.glob('pytorch_model*.bin'))):
    raise SystemExit(f'未找到模型权重：{model}')
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit(f'输出路径已被占用，请设置新的 OUTPUT_DIR：{output}')

import torch
import pandas as pd
import evaluate  # 检查实际评估入口的依赖。
from transformers import AutoTokenizer

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('需要且只能暴露一张 CUDA 显卡，请检查 CUDA_VISIBLE_DEVICES')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('当前评估代码需要 BF16 支持')
tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
indices = json.loads(index.read_text())
# 不现场扩词：checkpoint 必须已经保存正确的词表和对应模型权重。
for parts in indices.values():
    ids = [tokenizer.encode(part, add_special_tokens=False) for part in parts]
    if any(len(x) != 1 for x in ids):
        raise SystemExit('checkpoint tokenizer 未正确包含 SID token，请使用对应类别的 SFT/RL checkpoint')
    if tokenizer.encode(''.join(parts), add_special_tokens=False) != [x[0] for x in ids]:
        raise SystemExit('完整 SID 编码与部件编码不一致')
config = json.loads((model / 'config.json').read_text())
if config['vocab_size'] < len(tokenizer):
    raise SystemExit('模型词表小于 tokenizer 词表，请检查 checkpoint 是否完整')
df = pd.read_csv(test)
if len(df) < 32:
    raise SystemExit('测试集不足 32 条')
output.mkdir(parents=True, exist_ok=True)
df.head(32).to_csv(output / 'test-32.csv', index=False)
print(f'预检通过：GPU={torch.cuda.get_device_name(0)}，已生成 32 条测试子集')
PY

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/evaluate.log"

# 检查全部 320 个候选，避免 calc.py 命中目标后提前退出导致漏检。
"$PYTHON_BIN" - "$RESULT_FILE" "$INFO_FILE" "$SUBSET_FILE" <<'PY' 2>&1 | tee "$OUTPUT_DIR/validation.log"
import csv
import json
import sys
from pathlib import Path

result, info, subset = map(Path, sys.argv[1:])
rows = json.loads(result.read_text())
valid_sids = {line.split('\t')[0].strip() for line in info.read_text().splitlines() if line.strip()}
with subset.open(newline='', encoding='utf-8') as f:
    targets = [row['item_sid'].strip() for row in csv.DictReader(f)]
if not isinstance(rows, list) or len(rows) != 32 or len(targets) != 32:
    raise SystemExit('FAIL：结果或测试子集数量不是 32')
invalid = []
for i, (row, target) in enumerate(zip(rows, targets)):
    if not isinstance(row.get('output'), str) or row['output'].strip(' \n"') != target:
        raise SystemExit(f'FAIL：第 {i + 1} 条结果的目标与测试子集不一致')
    if target not in valid_sids:
        raise SystemExit(f'FAIL：第 {i + 1} 条目标不在 info 合法 SID 集合中')
    predictions = row.get('predict')
    if not isinstance(predictions, list) or len(predictions) != 10:
        raise SystemExit(f'FAIL：第 {i + 1} 条结果没有 10 个候选')
    for rank, sid in enumerate(predictions, start=1):
        if not isinstance(sid, str) or sid.strip(' \n"') not in valid_sids:
            invalid.append((i + 1, rank, sid))
print(f'样本：32，候选总数：320，非法候选：{len(invalid)}')
if invalid:
    print('非法候选示例：', invalid[:10])
    raise SystemExit('FAIL：存在非法 SID')
print('PASS：全部候选合法，结果与测试子集对齐')
PY

"$PYTHON_BIN" "$PROJECT_ROOT/calc.py" \
  --path "$RESULT_FILE" --item_path "$INFO_FILE" \
  2>&1 | tee "$OUTPUT_DIR/metrics.log"
echo "评估完成：$OUTPUT_DIR"
