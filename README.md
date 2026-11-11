# Hupkes compositionality reproduction: local training

This is a portable export of the original Hupkes code/data and our CPU training
worker. It does not require a cloud terminal. The full ordinary-task seed is
unfinished; this archive contains no completed reproduction result.

**macOS users: start with `START_HERE_MAC.md`.** It provides Docker Desktop
commands for Intel and Apple Silicon. The native Linux instructions below are
an alternative for Linux and WSL users.

## Supported computer

The tested legacy runtime is **x86_64 Linux**. On Windows, use Ubuntu in WSL2
and run these commands in the Ubuntu terminal. Keep the extracted folder in
your Linux home directory. Native Windows, macOS and ARM are not supported by
this particular pinned PyTorch 1.4 runtime natively. macOS can instead use the
included Docker Desktop workflow.
No GPU is used or required. About 32 GB RAM and 40 GB free disk are recommended
for this configuration and retained checkpoints. Smaller machines may need an
explicitly documented batch-size change. Training takes many hours or days;
prevent your computer from sleeping during a run. Internet is needed for setup.

## Install

On GitHub choose **Code → Download ZIP**, unzip it, then open Terminal in the extracted `RNTN-main` folder. Alternatively, clone this repository.
Install uv if it is not already available, using the official instructions:
https://docs.astral.sh/uv/getting-started/installation/

For example, when Python with pip is installed:

```bash
python3 -m pip install --user uv
```

If your Linux distribution blocks user pip installations, use its supported
`pipx install uv` route or another official uv installation method. Ensure
`uv --version` works before continuing.

```bash
bash setup.sh
bash smoke.sh
```

The smoke command independently checks the original datasets, trains a tiny
Transformer, requires four/four exact decoded predictions and resumes a saved
checkpoint. It is separate from the full training experiment.

## Train

```bash
.venv/bin/python train.py
```

Keep that terminal open. Training runs in the foreground. Logs, configuration,
dataset hashes and model checkpoints go into:

```text
training/hupkes-transformer-seed1/
```

To monitor it in a second terminal, from the same folder:

```bash
tail -f training/hupkes-transformer-seed1/train.log
```

Checkpoints are saved every640 updates; the target is32125. After training,
the worker selects the validation-best checkpoint, decodes the original9721
test examples and writes `results.json` and `test-predictions.txt`.
This evaluation happens automatically; its full-run outcome is still unverified.

## Resume after interruption

First make sure the previous training process has stopped. List checkpoints:

```bash
ls training/hupkes-transformer-seed1/model_step_*.pt
```

Use an actual filename from that list, for example if step640 exists:

```bash
.venv/bin/python train.py --resume training/hupkes-transformer-seed1/model_step_640.pt
```

Do not type the example checkpoint path unless that file exists. If no
checkpoint was saved, rerun `.venv/bin/python train.py` to restart from seed1.
Never run two workers writing into the same training directory.

## What was transferred

* Original Hupkes repository export, including all released datasets, MIT
  license and training/evaluation source, pinned at commit
  `4f75bdd78889150dd78e90bbb7ddec840204c159`.
* A portable worker and exact dependency versions: Python3.8.20,
  OpenNMT-py1.0.0, CPU torch1.4.0 and torchtext0.4.0.
* Our smoke/data checks and documented methodological assumptions.

No cloud training checkpoints are included in this repository. **This local run starts from scratch; the partially trained
weights were not transferred.** No cloud credentials, virtual environments,
Git authentication files or process IDs are included.

The local command trains the ordinary PCFG SET Transformer baseline only.
Additional compositionality experiments and Socher are not implemented by this
worker. Adam beta2=0.998 follows the released script, whereas paper section5.3.2
says0.98. Original preprocessing defaults retain81959 of82622 training pairs,
with all4860 dev examples. Read `REPRODUCTION_NOTES.md` for the source history
and limitations; any `/workspace` paths there refer to the cloud machine.

Cloud training is independent of this repository. Downloading it does not stop the
cloud worker or transfer a running process to your machine.
