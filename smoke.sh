#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export bundle_dir
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$bundle_dir/.cache" MPLCONFIGDIR="$bundle_dir/.cache/matplotlib"
bin="${HUPKES_BIN:-$bundle_dir/.venv/bin}"
mkdir -p "$bundle_dir/validation/hupkes"
run_dir=$(mktemp -d "$bundle_dir/validation/hupkes/smoke-XXXXXXXX")
export run_dir
"$bin/python" "$bundle_dir/validate_data.py"
"$bin/python" - <<'PY'
import os
from pathlib import Path
p=Path(os.environ['bundle_dir']) / 'original/data/pcfgset/pcfgset'
out=Path(os.environ['run_dir'])
pairs=list(zip((p/'train.src').read_text().splitlines(), (p/'train.tgt').read_text().splitlines()))
pairs=[(s,t) for s,t in pairs if len(s.split())<=4 and len(t.split())<=3][:4]
assert len(pairs)==4
for split in ['train','dev']:
    for i,ext in enumerate(['src','tgt']):
        (out/f'{split}.{ext}').write_text('\n'.join(x[i] for x in pairs)+'\n')
PY
"$bin/onmt_preprocess" -train_src "$run_dir/train.src" -train_tgt "$run_dir/train.tgt" \
  -valid_src "$run_dir/dev.src" -valid_tgt "$run_dir/dev.tgt" -save_data "$run_dir/data" > "$run_dir/preprocess.log" 2>&1
"$bin/onmt_train" -data "$run_dir/data" -save_model "$run_dir/tiny" \
  -layers 2 -rnn_size 64 -word_vec_size 64 -transformer_ff 128 -heads 4 \
  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0 \
  -batch_size 4 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 20 \
  -learning_rate 1 -max_grad_norm 0 -param_init 0 -param_init_glorot \
  -valid_steps 100 -train_steps 300 -save_checkpoint_steps 300 -report_every 100 -seed 1 > "$run_dir/train.log" 2>&1
"$bin/onmt_translate" -model "$run_dir/tiny_step_300.pt" -src "$run_dir/dev.src" \
  -output "$run_dir/pred.txt" -batch_size 4 -beam_size 5 -max_length 20 > "$run_dir/decode.log" 2>&1
"$bin/onmt_train" -data "$run_dir/data" -train_from "$run_dir/tiny_step_300.pt" \
  -save_model "$run_dir/resumed" -train_steps 301 -save_checkpoint_steps 301 \
  -valid_steps 301 -report_every 1 > "$run_dir/resume.log" 2>&1
"$bin/python" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['run_dir'])
target=(p/'dev.tgt').read_text().splitlines()
prediction=(p/'pred.txt').read_text().splitlines()
assert len(target)==len(prediction)==4
assert target==prediction, (target,prediction)
assert (p/'resumed_step_301.pt').exists()
result={'purpose':'environment smoke; not a paper reproduction', 'seed':1, 'tiny_examples':4, 'exact_matches':4, 'resume_step':301}
(p/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
print('Artifacts:',p)
PY
