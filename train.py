"""Launch/resume the first original-script Hupkes Transformer baseline on CPU.

This worker persists state on disk; it does not provision compute or keep a cloud
environment awake. Its parent may detach it with subprocess.Popen.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import re
import subprocess
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'training' / 'hupkes-transformer-seed1'
DATA = BASE / 'original' / 'data' / 'pcfgset' / 'pcfgset'
BIN = Path(sys.executable).parent
STATE = ROOT / 'status.json'


def state(phase, **kwargs):
    record = {'phase': phase, 'worker_pid': os.getpid(), 'updated_unix': time.time(), **kwargs}
    temp = ROOT / 'status.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(STATE)


def command(args, log):
    with (ROOT / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(args) + '\n')
    with (ROOT / log).open('a', buffering=1) as stream:
        subprocess.run(args, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      XDG_CACHE_HOME=str(BASE/'.cache'), MPLCONFIGDIR=str(BASE/'.cache/matplotlib'),
                      PYTHONPYCACHEPREFIX=str(BASE/'.cache/pycache'))
    if not args.resume and list(ROOT.glob('model_step_*.pt')):
        raise RuntimeError('Existing checkpoints: resume explicitly rather than overwrite them')
    if args.resume and (args.resume.resolve().parent != ROOT or not args.resume.exists()):
        raise RuntimeError('Resume requires an existing checkpoint in this run directory')
    state('preflight', resumed_from=str(args.resume) if args.resume else None)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in DATA.glob('*') if p.suffix in {'.src', '.tgt'}}
    metadata = {
        'purpose': 'First full ordinary-task Transformer seed; reproduction not yet established',
        'source_commit': '4f75bdd78889150dd78e90bbb7ddec840204c159',
        'seed': 1, 'device': 'cpu', 'threads': 4, 'steps': 32125,
        'layers': 6, 'dimension': 512, 'feedforward': 2048, 'heads': 8,
        'batch_size': 64, 'dropout': 0.1, 'warmup_steps': 8000,
        'adam_beta1': 0.9, 'adam_beta2': 0.998,
        'data_sha256': hashes, 'validation_steps': 1280,
        'checkpoint_steps': 640, 'checkpoint_selection': 'highest validation token accuracy, as released script',
        'departures_and_assumptions': [
            'Paper section 5.3.2 says beta2=0.98; released script uses 0.998. Follow released script.',
            'CPU Python3.8/OpenNMT1.0/torch1.4 compatibility runtime; exact original toolkit unpinned.',
            'Preprocessing defaults: train source/target length limit 50, matching released template defaults.',
            'New seed 1; paper uses three random seeds but does not identify their values.',
            '32125 updates follow released ordinary-task script; not an exact 25-epoch count.',
            'Checkpoint every640 instead of1280 for recovery; validation interval remains1280.',
            'Seed run is not evidence for other compositionality tests; those remain pending.',
            'Decode beam5 and max_length300 explicitly chosen; exact original decoding defaults unverified.',
        ],
    }
    (ROOT / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if not (ROOT / 'data.vocab.pt').exists():
        command([str(BIN/'onmt_preprocess'), '-train_src', str(DATA/'train.src'),
                 '-train_tgt', str(DATA/'train.tgt'), '-valid_src', str(DATA/'dev.src'),
                 '-valid_tgt', str(DATA/'dev.tgt'), '-save_data', str(ROOT/'data')], 'preprocess.log')
    train = [str(BIN/'onmt_train'), '-data', str(ROOT/'data'), '-save_model', str(ROOT/'model'),
             '-layers', '6', '-rnn_size', '512', '-word_vec_size', '512', '-transformer_ff', '2048',
             '-heads', '8', '-encoder_type', 'transformer', '-decoder_type', 'transformer',
             '-position_encoding', '-max_generator_batches', '2', '-dropout', '0.1',
             '-batch_size', '64', '-optim', 'adam', '-adam_beta2', '0.998', '-decay_method', 'noam',
             '-warmup_steps', '8000', '-learning_rate', '1', '-max_grad_norm', '0', '-param_init', '0',
             '-param_init_glorot', '-valid_steps', '1280', '-train_steps', '32125',
             '-save_checkpoint_steps', '640', '-report_every', '1', '-seed', '1']
    if args.resume:
        train += ['-train_from', str(args.resume)]
    state('training', resumed_from=str(args.resume) if args.resume else None, target_steps=32125)
    command(train, 'train.log')
    # Associate each validation event with the actual next checkpoint path,
    # avoiding the released select_model.py's zero-based off-by-one index.
    candidates, pending = [], None
    for line in (ROOT/'train.log').read_text().splitlines():
        match = re.search(r'Validation accuracy: ([0-9.eE+-]+)', line)
        if match:
            pending = float(match.group(1))
        match = re.search(r'Saving checkpoint (.+_step_(\d+)\.pt)', line)
        if match and pending is not None:
            checkpoint = Path(match.group(1))
            if checkpoint.exists():
                candidates.append((pending, checkpoint))
            pending = None
    if not candidates:
        raise RuntimeError('Training ended without a validated checkpoint')
    best_accuracy, best = max(candidates, key=lambda item: item[0])
    state('evaluating', best_checkpoint=str(best), validation_token_accuracy=best_accuracy)
    command([str(BIN/'onmt_translate'), '-model', str(best), '-src', str(DATA/'test.src'),
             '-output', str(ROOT/'test-predictions.txt'), '-batch_size', '16',
             '-beam_size', '5', '-max_length', '300'], 'test-decode.log')
    targets = (DATA/'test.tgt').read_text().splitlines()
    predictions = (ROOT/'test-predictions.txt').read_text().splitlines()
    if len(predictions) != len(targets) or not targets:
        raise RuntimeError('Missing or incomplete test predictions')
    correct = sum(p.strip() == t.strip() for p, t in zip(predictions, targets))
    result = {'seed': 1, 'best_checkpoint': str(best), 'validation_token_accuracy_percent': best_accuracy,
              'test_examples': len(targets), 'test_exact_matches': correct,
              'test_exact_match': correct/len(targets), 'paper_three_seed_mean': 0.92,
              'paper_three_seed_sd': 0.01, 'comparison_caveat': 'Single run, compatibility environment, documented assumptions'}
    (ROOT/'results.json').write_text(json.dumps(result, indent=2) + '\n')
    state('completed', **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('failed', error=str(exc))
        traceback.print_exc()
        raise
