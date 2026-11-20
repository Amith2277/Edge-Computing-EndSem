# AdaptivePark

Adaptive parking-occupancy pipeline for edge devices. See `../ADAPTIVEPARK_PLAN.md` for the full plan.

## Machines and roles

| Machine | Role |
|---|---|
| Laptop (Windows, no GPU) | Development, data inspection/conversion, occupancy + adaptive logic, evaluation scripts |
| Kaggle (GPU) | Fine-tune YOLO11n / YOLO11s |
| Jetson AGX Orin (JetPack 5.1.2, Python 3.8.10) | TensorRT export, real-time pipeline, latency/power benchmarks |

## Rules that keep the three machines compatible

1. **All code must run on Python 3.8** (the Orin's version), even though the laptop venv is 3.10.
   - No `match` statements, no `X | Y` type unions, no `list[int]` / `dict[str, int]` annotations unless the file starts with `from __future__ import annotations`.
   - Prefer `typing.List`, `typing.Optional`, etc.
2. Use `pathlib` for paths; never hard-code Windows or Linux paths. Paths live in `configs/*.yaml`.
3. The dataset size is a config value (`subset_fraction`), not a separate copy of the code.
4. Keep the same `ultralytics` version on Kaggle and the Orin so weights load everywhere (version to be fixed after testing on the Orin).
5. TensorRT engines are built **on the Orin** from the `.pt`/ONNX weights. Never copy engines between machines.

## Layout (planned)

```
code/
  configs/      # dataset paths, subset_fraction, thresholds
  src/          # dataset, detector, occupancy, features, difficulty, controller, temporal, drift, pipeline
  scripts/      # train, evaluate, benchmark entry points
  results/      # small CSV/JSON results (committed)
  models/       # weights (not committed)
  data/         # datasets (not committed)
```

## Laptop environment

Python 3.10 virtual environment in `.venv/`:

```bash
python -m venv .venv            # run with a Python 3.10 interpreter
.venv\Scripts\activate          # Windows
pip install -r requirements-laptop.txt
```
