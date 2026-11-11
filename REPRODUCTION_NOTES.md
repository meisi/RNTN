# Environment setup and reproduction status

## Current training update

Paper access is now working. The JAIR PDF was downloaded to
`/workspace/research/hupkes2020.pdf`; its extracted text is alongside it. Sections
4.3, 5.3.2, 6 and Appendix A were inspected for this training run. The entire
paper has not yet been reviewed in detail for all remaining evaluation tasks.

The first ordinary-task Transformer seed (seed 1) is now running under
`/workspace/training/hupkes-transformer-seed1`. Its `config.json` records data
hashes, settings, and departures; `commands.jsonl` records executable commands;
`train.log` records loss/accuracy each update; `status.json` records worker phase.
It targets 32125 updates, validates every1280 and saves checkpoints every640.
This uses the full original training corpus as input, with the toolkit's default
length filter inherited from the released script: 81959 of 82622 training pairs
are retained; all4860 validation examples are retained. This is not a tiny smoke
experiment. It is also not yet a successful reproduction.

Important recovered discrepancy: paper section5.3.2 specifies Adam beta2=0.98;
the authors' released script specifies0.998. This first seed follows the released
script at0.998. The exact original toolkit version and decoding defaults remain
ambiguous. CPU compatibility runtime, seed1 and more frequent640-step recovery
checkpoints are documented assumptions, not tuning choices based on test scores.

After training, the worker selects the checkpoint with highest validation token
accuracy, as the authors' selection script intends, then decodes the untouched
9721-example test set and writes sequence exact-match results. The worker fixes
the released selection script's zero-based checkpoint-index error without
editing upstream. That final full-run evaluation has not yet executed.

Processes only run while this cloud environment stays active. Publishing or
restoring a filesystem does not preserve a running process. Inspect current
state before launching/resuming; never start two workers for the same run.

The initial setup record below is retained as history; its paper-access blocker
and statement that no full run was started are superseded by this update.

This is setup evidence, not a completed reproduction of either paper. The request
is preserved in the uploaded attachment. Hupkes must be completed before Socher
implementation starts.

## Sources recovered

* Hupkes et al. (2020), *Compositionality decomposed: how do neural networks
  generalise?*, JAIR 67, 757–795.
  Paper link from the authors' README:
  https://jair.org/index.php/jair/article/view/11674
* Authors' code and original naturalised datasets:
  https://github.com/i-machine-think/am-i-compositional,
  commit `4f75bdd78889150dd78e90bbb7ddec840204c159`, MIT license.
* Generator linked from the authors' dataset README:
  https://github.com/MathijsMul/pcfg-set,
  commit `99098bd97150671003de60526d884d7c24949944`.
  No license file was found; do not assume the main repository's MIT license
  extends to the generator.
* The selected `meisi/RNTN` checkout is genuinely unborn: symbolic `work` branch,
  no refs or HEAD commit, no source files. Read-only origin access succeeded and
  returned no remote refs. No commit or branch was manufactured during setup.

## Blocked source verification

The full paper and appendices have NOT been read. The paper request failed with
`CONNECT tunnel failed, response 403` under the current restricted network
policy. The draft proposes `jair.org`, `www.jair.org`, `arxiv.org`,
`export.arxiv.org`, `aclanthology.org`, and `nlp.stanford.edu` for paper and
subsequent original SST access. These additions are saved, not yet proven active.
Review and save in environment settings; retry with a fresh connection before
relying on them. No paper numbers have been guessed or copied from a summary.

Socher source, license, full SST data, exact neutral handling and diagnostics
remain unverified. No Socher implementation or training has started.

## Recovered Hupkes workflow (repository evidence only)

The original flat, space-separated `.src` inputs and corresponding `.tgt` outputs
are retained unchanged. No gold trees or executor outputs are fed into training.
The common development set is `data/pcfgset/pcfgset/dev.{src,tgt}`.

The OpenNMT Transformer script specifies six encoder/decoder layers, model and
embedding dimension 512, feed-forward dimension 2048, eight attention heads,
positional encoding, dropout 0.1, batch size 64, Adam beta2 0.998, Noam schedule,
8000 warmup steps, learning-rate factor 1, Xavier initialization, no gradient
clipping, nominally 25 epochs, and validation-based checkpoint selection.
Its ordinary-task target is 32125 steps, validation/checkpoint interval 1280.
Those counts are not exactly 25*1280; this discrepancy needs paper verification.

The published shell files are templates, not immediately runnable commands:
they overwrite PATH with `...`, request unavailable GPUs, refer to
`systematicity/train2` despite the shipped `train` filename, and place common dev
files at a different path. The Transformer loop says three runs in a comment
but actually iterates only over `2`. Toolkit versions are not pinned;
`consistency.py` mentions November 2018 OpenNMT. Seed protocol, decoding details,
length filtering, exact toolkit revision, and paper-versus-code discrepancies
remain to be resolved from the paper before full experiments.

The LSTM script and Fairseq convolutional script are retained as sources;
Fairseq is not installed or validated. No modern tuned architecture substituted.

Metric definitions recovered from the authors' READMEs and scripts:

* Ordinary task, systematicity, productivity: sequence exact match on their
  original respective splits. Productivity additionally studies length/EOS.
* Substitutivity: compare original and synonym-replaced predictions, in BOTH
  equally-distributed and primitive training conditions. The consistency script
  counts matching nonempty predictions, even when wrong; not just accuracy.
* Localism: compare whole-expression model predictions against recursively
  unrolled MODEL predictions, propagating model errors through placeholders.
  Gold execution must not replace these intermediate model predictions.
* Overgeneralisation: track original-rule versus exceptional-target predictions
  over training, then average per-run peak heights. The exception ratios shipped
  are 0.0001, 0.0005, 0.001, 0.005. This is not a final-test accuracy metric.

## Runtime, changes and limitations

Machine: 4 CPU cores of cgroup quota, about 33 GiB memory limit, about 60 GiB free
disk at setup; no exposed NVIDIA devices or CUDA. Host Python is 3.12.14.
No drivers, paid resources, secrets, or external services were configured.

An isolated legacy environment at `/workspace/.venvs/hupkes` uses CPython 3.8.20,
OpenNMT-py 1.0.0, CPU PyTorch 1.4.0, torchtext 0.4.0, NumPy 1.24.4, and the exact
dependencies in `runtime-requirements.txt`. This is an explicit compatibility
choice, NOT the verified original environment and NOT a current-PyTorch port.
Use it for the trusted original sources; it is isolated from host Python.

The authors' custom translation scripts use a different toolkit API:
their `translate.py` calls `translate(src_path=..., tgt_path=...)`, whereas the
installed API takes `src` and `tgt`. Their localism script requires
`Translator.translate_one`, absent from this release. These require an explicit
port or recovery of the matching toolkit modifications in the reproduction
coding phase. Core OpenNMT preprocessing/training/decoding works; the entire
five-test neural evaluation pipeline is NOT validated.

Upstream generator needs explicit `import tasks.default` before its top-level
imports. `generator_smoke.py` supplies that import without editing upstream;
the generated 100 examples are only a generator check, NOT a replacement for
the original naturalised splits. Re-running naturalisation would require the
original WMT distribution, CoreNLP and recovered generation settings. Existing
original datasets are available, so these optional services were not started.

Smoke departures: first 64 training / 16 dev examples, CPU, two training steps,
explicit preprocessing length limits of 1000 to retain these examples. This
pilot retains the script's 6x512 Transformer architecture. Its vocabulary is
built only on that tiny subset. The separate overfit smoke uses four short real
training examples, two layers, dimension 64, FF dimension 128, four heads,
dropout zero, 20 warmup steps, 300 updates, seed 1. Training examples reused as
tiny validation data intentionally test memorization, never generalization.

The pilot measured about 26 seconds for two training updates (four threads).
At 13 seconds/update, 32125 updates are roughly 4.8 days per baseline seed.
This is only a rough extrapolation: full vocabulary, length mix, data loading,
validation and other experiment conditions will change timing. No full run was
started before source/method verification. No full-seed result exists.

Tracked source, tests, manifests and lockfiles in the checkouts remain unchanged.
Setup helpers, environment versions, logs and generated checkpoints live outside
them, in `/workspace/setup`, `/workspace/.venvs`, and `/workspace/validation`.

## Required next work

1. Apply paper/data network settings and verify fresh access.
2. Read both full papers and appendices; extract the exact reference metrics and
   recover missing hyperparameters and seeds without guessing.
3. Port the custom toolkit interfaces and validate all five neural evaluation
   pipelines, preserving their metric definitions.
4. Run one full Hupkes seed after method and runtime validation, retaining all
   predictions and curves; choose additional seeds based on measured cost.
5. Complete/document Hupkes before implementing or training Socher.
