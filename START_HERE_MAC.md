# Run on macOS

1. Install and open **Docker Desktop for Mac**:
   https://www.docker.com/products/docker-desktop/
   Choose the installer matching Apple Silicon or Intel. This uses local CPU
   compute, not a cloud service. Make sure Docker Desktop is running.
2. On GitHub choose **Code → Download ZIP**, then unzip it. Open the macOS **Terminal** app (Spotlight: Cmd+Space,
   type Terminal). Type `cd ` with a trailing space, drag the extracted
   `RNTN-main` folder into Terminal, then press Return.
3. Run these commands one at a time:

```bash
bash macos.sh build
bash macos.sh smoke
bash macos.sh train
```

The first command downloads/builds the legacy runtime. The second should report
four/four exact matches and a resumed step301. The third starts the actual
32125-step ordinary-task Transformer training. Leave Terminal and Docker
Desktop open, and prevent your Mac from sleeping while training.

Results and checkpoints are saved on your Mac under
`RNTN-main/training/hupkes-transformer-seed1/`. This bind-mounted directory
survives container exit. Do not move the folder or launch a duplicate worker
while training runs. You can read `train.log` in a text editor.

The CPU-only legacy PyTorch runtime is x86_64 Linux inside Docker. Intel Macs
run it through the Linux VM; Apple Silicon also needs x86 emulation, which can
be significantly slower. Docker Desktop must support `linux/amd64` emulation.
Check Docker Desktop Settings → Resources: allow at least8GB RAM if your Mac
has enough memory;16GB is preferable for full training. Keep at least40GB disk
free for dependencies and saved checkpoints. A Mac with only8GB total RAM may
need a smaller batch or a separate native implementation.

Training can take days, particularly on Apple Silicon under emulation. This
package does not use Apple's GPU/MPS. A native modern-PyTorch port is separate
work and would need to be checked against the original implementation.

To resume after stopping, first ensure the previous container is stopped. Use
an actual checkpoint filename saved on your Mac, for example:

```bash
bash macos.sh resume training/hupkes-transformer-seed1/model_step_640.pt
```

If that file does not exist, choose one that does; if there is no checkpoint,
start again with `bash macos.sh train`.

The cloud run had no saved checkpoint when this export was made. Your local
run therefore starts fresh. Full Hupkes reproduction and the other paper's
experiments remain incomplete; a successful smoke test is not a paper result.
