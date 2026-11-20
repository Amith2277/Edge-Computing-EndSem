# Jetson AGX Orin setup notes

Target: Jetson AGX Orin Developer Kit, JetPack 5.1.2 (L4T 35.4.1), Ubuntu 20.04, **Python 3.8.10**, CUDA 11.4, aarch64.

> Commands here are a starting point and **have not been verified on this board yet**. Run Step 1 first and report the output; later steps may change based on it.

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

## Step 2: Python environment

Use a venv that can see the system packages (TensorRT bindings come from apt):

```bash
python3 -m venv --system-site-packages ~/adaptivepark-env
source ~/adaptivepark-env/bin/activate
pip install --upgrade pip
```

## Step 3: PyTorch for JetPack 5.1.2

Plain `pip install torch` gives a CPU-only aarch64 build. Use NVIDIA's JetPack wheel (cp38, JetPack 5.1.x). Check NVIDIA's "PyTorch for Jetson" page for the exact wheel name for JetPack 5.1.2 before downloading. `torchvision` normally has to be built from source to match the torch version.

## Step 4: ultralytics

Install the version we pin for all machines (TBD, must be >= 8.3.0 for YOLO11 and compatible with the torch above). Test that a `.pt` file trained on Kaggle loads and runs here.

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
