#!/usr/bin/env bash
# MODEL_PATH=/path/to/sft/final_checkpoint bash rl_smoke_autodl.sh [--dry-run]
# 仅用于链路验证：add_gt=True 会把真实答案加入候选组，不是正式实验配置。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "用法：MODEL_PATH=/path/to/sft/final_checkpoint bash rl_smoke_autodl.sh [--dry-run]" >&2
  exit 2
fi
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "请设置 MODEL_PATH，指向 SFT 生成的 final_checkpoint。" >&2
  exit 2
fi
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"
case "$CATEGORY" in
  Industrial_and_Scientific|Office_Products) ;;
  *) echo "此脚本仅支持 Industrial_and_Scientific / Office_Products" >&2; exit 2 ;;
esac
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/rl-smoke-${CATEGORY}-$(date +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

TRAIN_FILE="$PROJECT_ROOT/data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="$PROJECT_ROOT/data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="$PROJECT_ROOT/data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
INDEX_FILE="$PROJECT_ROOT/data/Amazon/index/${CATEGORY}.index.json"
ITEM_FILE="$PROJECT_ROOT/data/Amazon/index/${CATEGORY}.item.json"

# 单进程运行；不读取 Accelerate 启动配置，也不启用 DeepSpeed。
# 全局训练/验证 batch 均为 4，可被 num_generations=4 整除。
command=(
  "$PYTHON_BIN" "$PROJECT_ROOT/rl.py"
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
  --train_file "$TRAIN_FILE"
  --eval_file "$EVAL_FILE"
  --info_file "$INFO_FILE"
  --sid_index_path "$INDEX_FILE"
  --item_meta_path "$ITEM_FILE"
  --category "$CATEGORY"
  --debug_sample 64
  --max_steps 5
  --max_completion_length 16
  --train_batch_size 4
  --eval_batch_size 4
  --gradient_accumulation_steps 1
  --num_generations 4
  --num_train_epochs 1
  --eval_step 0.5
  --learning_rate 1e-6
  --temperature 1.0
  --beta 0.001
  --reward_type ranking
  --add_gt True
  --beam_search False
  --dynamic_sampling False
  --sync_ref_model False
  --test_during_training False
  --sample_train False
  --optim adamw_torch
  --report_to none
  --save_total_limit 1
  --bf16 True
  --fp16 False
  --dapo False
  --gspo False
  --seed 42
)
printf '执行命令：\n'
printf '%q ' "${command[@]}"
printf '\n输出目录：%s\n' "$OUTPUT_DIR"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "仅预览；未检查 checkpoint、依赖或 GPU，未创建输出目录。"
  exit 0
fi
if [[ "${WORLD_SIZE:-1}" != "1" || "${ACCELERATE_USE_DEEPSPEED:-false}" == "true" ]]; then
  echo "请在普通单进程终端中启动此脚本，不要使用多卡或 DeepSpeed 启动器。" >&2
  exit 1
fi

"$PYTHON_BIN" - "$MODEL_PATH" "$TRAIN_FILE" "$EVAL_FILE" "$INFO_FILE" "$INDEX_FILE" "$ITEM_FILE" "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

model, train, valid, info, index, item, output = map(Path, sys.argv[1:])
for path in (model / 'config.json', train, valid, info, index, item):
    if not path.is_file():
        raise SystemExit(f'缺少文件：{path}')
if not (list(model.glob('*.safetensors')) or list(model.glob('pytorch_model*.bin'))):
    raise SystemExit(f'未找到模型权重：{model}')
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise SystemExit(f'输出路径已被占用，请设置新的 OUTPUT_DIR：{output}')
for path in (train, valid):
    with path.open(newline='', encoding='utf-8') as f:
        if sum(1 for _ in csv.DictReader(f)) < 64:
            raise SystemExit(f'数据不足 64 条：{path}')

import torch
import rl  # 检查自定义 Trainer 和 TRL 等实际导入依赖。
from transformers import AutoTokenizer

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('需要且只能暴露一张 CUDA 显卡，请检查 CUDA_VISIBLE_DEVICES')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('此脚本需要 BF16 支持')
tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
indices = json.loads(index.read_text())
legal_sids = {line.split('\t')[0].strip() for line in info.read_text().splitlines() if line.strip()}
if {''.join(parts) for parts in indices.values()} != legal_sids:
    raise SystemExit('info 与 index 的 SID 集合不一致')
for parts in indices.values():
    ids = [tokenizer.encode(part, add_special_tokens=False) for part in parts]
    if any(len(x) != 1 for x in ids):
        raise SystemExit('checkpoint tokenizer 缺少独立 SID token，请使用对应类别的 SFT checkpoint')
    if tokenizer.encode(''.join(parts), add_special_tokens=False) != [x[0] for x in ids]:
        raise SystemExit('完整 SID 编码与部件编码不一致')
if json.loads((model / 'config.json').read_text())['vocab_size'] < len(tokenizer):
    raise SystemExit('模型词表小于 tokenizer 词表')
print(f'预检通过：torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}')
PY

mkdir -p "$OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"

# 只核实完成步数、日志数值和导出文件；不将其等同于模型效果提升。
"$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY' 2>&1 | tee "$OUTPUT_DIR/validation.log"
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
states = list(output.glob('checkpoint-*/trainer_state.json'))
if not states:
    raise SystemExit('FAIL：没有找到 Trainer 状态文件')
state = max((json.loads(p.read_text()) for p in states), key=lambda x: x['global_step'])
if state['global_step'] != 5:
    raise SystemExit(f"FAIL：只完成 {state['global_step']} 步，预期 5 步")
logs = state.get('log_history', [])
for entry in logs:
    for key, value in entry.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise SystemExit(f'FAIL：日志中 {key} 出现非有限值：{value}')
final = output / 'final_checkpoint'
if not (final / 'config.json').is_file() or not (final / 'tokenizer_config.json').is_file():
    raise SystemExit('FAIL：最终模型或 tokenizer 配置缺失')
if not (list(final.glob('*.safetensors')) or list(final.glob('pytorch_model*.bin'))):
    raise SystemExit('FAIL：最终模型权重缺失')
print('PASS：完成 5 步，已记录的数值有限，最终 checkpoint 文件存在')
for key in ('reward', 'reward_std', 'grad_norm', 'kl'):
    print(f'{key}:', [entry[key] for entry in logs if key in entry])
print('请继续用 eval_smoke_autodl.sh 验证最终模型能重新加载和生成。')
PY
echo "RL 试跑结束：$OUTPUT_DIR/final_checkpoint"
