# Jetson AGX Orin setup notes

Target: Jetson AGX Orin Developer Kit, JetPack 5.1.2 (L4T 35.4.1), Ubuntu 20.04, **Python 3.8.10**, CUDA 11.4, aarch64.

> Commands here are a starting point and **have not been verified on this board yet**. Run Step 1 first and report the output; later steps may change based on it.

Memory: the lab board is the **64 GB** AGX Orin (confirmed by the user). The project spec text says 32 GB; the spec is outdated on this point. Every benchmark report must state the actual module and memory (64 GB).

The edge-side workflow (environment check, transfer with hash verification, PyTorch reference, ONNX parity, TensorRT FP16, full replay, sustained run, report) is defined in spec Part J and implemented by the scripts in `scripts/` (`check_jetson_environment.py`, `export_onnx.py`, `validate_onnx.py`, `build_tensorrt.py`, `benchmark_jetson.py`, `collect_jetson_telemetry.py`, `run_replay.py`, `generate_jetson_report.py`). This file only covers getting the board ready. Code run here must be Python 3.8-compatible; the backend (FastAPI/Pydantic) versions on Python 3.8 are not yet verified.

## Step 0: Put the NVMe drive to use (storage)

`lsblk` shows a 465.8 GB `nvme0n1` with **no partitions and no mount point**, while the eMMC root has only about 7.9 GB free. Datasets, ONNX files, TensorRT engines and logs should all live on the NVMe.

**Warning: `mkfs` erases everything on the drive.** Only continue if the checks below show it is empty. Run them first and read the output:

```bash
lsblk -f /dev/nvme0n1          # no FSTYPE and no partitions = unused
sudo blkid /dev/nvme0n1        # no output = no filesystem signature
sudo wipefs -n /dev/nvme0n1    # dry run: lists signatures, erases nothing
```

If **any** of them shows a filesystem signature (e.g. `ext4`), the drive is **not blank**. Do not format yet: mount it read-only and look at what is on it first (`sudo mkdir -p /mnt/check && sudo mount -o ro /dev/nvme0n1 /mnt/check && ls /mnt/check`).

If, and only if, all three show nothing (or you have checked the contents and they are not needed), partition, format and mount it:

> Status: done on the Orin on 2026-11-20. The drive had an ext4 filesystem on the whole disk before formatting, so its previous contents were erased. It is now mounted at `/mnt/nvme` (435 GB free) via an `/etc/fstab` entry with `nofail`.

```bash
sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L adaptivepark /dev/nvme0n1p1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1p1 /mnt/nvme
sudo chown -R $USER:$USER /mnt/nvme
echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p1) /mnt/nvme ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a && df -h /mnt/nvme
```

Then use `/mnt/nvme/adaptivepark/` for the repo clone, datasets (`data/`), weights and engines. The Python venv (`~/adaptivepark-env`) can stay on the eMMC; it is small.

## Step 1: Diagnose what is already installed

`jetson_release` reported "TensorRT: Not installed". JetPack 5.1.2 normally ships TensorRT 8.5.x, so this may just be a missing package or a reporting quirk. Run:

```bash
dpkg -l | grep -iE "nvinfer|tensorrt" | head -20
ls /usr/src/tensorrt/bin/trtexec 2>&1
python3 -c "import tensorrt; print(tensorrt.__version__)" 2>&1
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1
pip3 --version
df -h ~
sudo nvpmodel -q
```

If TensorRT is missing, the usual fix is `sudo apt update && sudo apt install nvidia-jetpack` (installs the full JetPack component set). Do this only after reviewing what Step 1 shows.

### Step 1 results (reported from the Orin)

- TensorRT **8.5.2.2 is installed and importable** (`python3 -c "import tensorrt"` works; `trtexec` at `/usr/src/tensorrt/bin/trtexec`). The "Not installed" line from `jetson_release` was wrong. No `apt install nvidia-jetpack` needed.
- `torch` is **not installed** yet.
- pip 25.0.1 on Python 3.8, venv `~/adaptivepark-env` created with `--system-site-packages`.
- Power mode is MAXN.
- **Disk is tight: the root filesystem (eMMC, 57 GB) has only 8.7 GB free (84% used).** An unused 465.8 GB NVMe (`nvme0n1`) is present; see Step 0.

## Step 2: Python environment

Use a venv that can see the system packages (TensorRT bindings come from apt):

```bash
python3 -m venv --system-site-packages ~/adaptivepark-env
source ~/adaptivepark-env/bin/activate
pip install --upgrade pip
```

## Step 3: PyTorch for JetPack 5.1.2

Plain `pip install torch` gives a CPU-only aarch64 build. Use NVIDIA's JetPack wheel (cp38, JetPack 5.1.x, folder `v512`). The wheel I expect is `torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl` under `https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/`, **but confirm the exact filename on NVIDIA's "PyTorch for Jetson" page before downloading** (I could not verify it from the laptop).

```bash
source ~/adaptivepark-env/bin/activate
sudo apt install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev libpng-dev
pip install "numpy<1.24"
pip install --no-cache-dir <path-or-url-of-the-torch-wheel>
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```

`torchvision` must be built from source. **Use v0.15.1, not v0.16.1:** NVIDIA's `2.1.0a0+41361538.nv23.06` wheel is an early 2.1 pre-release without `torch._custom_ops`, so a 0.16.1 build fails on import. The build takes 20+ minutes.

```bash
pip uninstall -y torchvision    # only if a wrong version is installed
cd /mnt/nvme/adaptivepark && git clone --branch v0.15.1 --depth 1 https://github.com/pytorch/vision torchvision-0151 && cd torchvision-0151
export BUILD_VERSION=0.15.1 TORCH_CUDA_ARCH_LIST="8.7" && python3 setup.py install
cd ~ && python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available()); from torchvision.ops import nms; print('ops ok')"
```

Before any install, send temp and cache files to the NVMe (eMMC space is scarce). These exports only last for the current terminal, so repeat them in each new one:

```bash
source ~/adaptivepark-env/bin/activate && mkdir -p /mnt/nvme/tmp /mnt/nvme/adaptivepark && export TMPDIR=/mnt/nvme/tmp PIP_CACHE_DIR=/mnt/nvme/pipcache
```

If `mkdir` under `/mnt/nvme` is denied, check `findmnt /mnt/nvme` (must show `/dev/nvme0n1p1`; otherwise `sudo mount -a`) and `ls -ld /mnt/nvme` (owner must be you; otherwise `sudo chown -R $USER:$USER /mnt/nvme`).

> Verified on the Orin on 2026-11-23: torch `2.1.0a0+41361538.nv23.06`, torchvision `0.15.1`, CUDA `True`, `nms` works.

## Step 4: ultralytics

Install the pinned version **after** torch and torchvision, so pip does not replace NVIDIA's torch with a PyPI build:

```bash
pip install --no-cache-dir "ultralytics==8.3.40"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must still show the nv wheel and True
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.yaml'); print('ok')"
```

If pip tries to download a new torch, stop and use `pip install --no-deps ultralytics==8.3.40`, then install its other dependencies by hand.

> Verified on the Orin on 2026-11-23: `ultralytics 8.3.40` installed without replacing torch. `YOLO('yolo11n.pt').predict(ASSETS/'bus.jpg', device=0)` found 5 boxes (4 people, 1 bus); first call took about 230 ms inference including warm-up (not a benchmark).
>
> Known harmless warning: `torchvision==0.15 is incompatible with torch==2.1`. It comes from ultralytics' own version table; the pair works.
>
> Network: downloading `bus.jpg` from `ultralytics.com` failed with a connection reset, while `yolo11n.pt` from GitHub worked. Use the bundled image `ultralytics.utils.ASSETS / 'bus.jpg'` for tests and copy datasets/weights over instead of relying on downloads.
>
> OpenCV: `pip` installed `opencv-python` 5.0.0.93 into `~/.local`, so `import cv2` gives 5.0.0 (not the system 4.5.4). It works with ultralytics so far. Note this version in every experiment's environment record.
>
> **Still unverified:** that a `.pt` trained on Kaggle (same ultralytics 8.3.40) loads and runs here. Test as soon as the first checkpoint exists.

## Step 5: Power modes for benchmarks

```bash
sudo nvpmodel -q              # current mode (MAXN reported)
sudo nvpmodel -m <mode_id>    # switch; list modes in /etc/nvpmodel.conf
sudo jetson_clocks            # lock clocks for stable timing
tegrastats                    # power / utilization while running
```

Benchmark MAXN **and** at least one constrained mode, and record which mode each result used.

## Handoff convention

The laptop session writes code and benchmark scripts into this repo. The Orin session pulls the repo, runs them, and commits small result files (CSV/JSON) into `results/`. Weights are copied directly, not through git.
