# AdaptivePark

Context-aware adaptive edge AI for parking occupancy detection. The system decides how much inference to run (YOLO11n or YOLO11s) from scene difficulty, per-slot transition risk and observation age, and is measured on a real Jetson AGX Orin.

Authoritative requirements: `AdaptivePark_Complete_Project_Specification_and_Master_Prompt_Updated (2).md` (the spec). Task list and plan: `../ADAPTIVEPARK_PLAN.md`.

## Machines and roles

| Machine | Term | Role |
|---|---|---|
| Laptop (Windows, no GPU) | Development workstation | Write code, unit tests, git. No experiments. |
| Kaggle GPU notebooks | Cloud training / offline evaluation | Dataset audit, splits, YOLO training, detection/occupancy/temporal evaluation, D/R tuning, controller replay, baselines, ablations, plots. **Offline only: no inference is served from the cloud.** Kaggle timings are never reported as Jetson timings. |
| Jetson AGX Orin (JetPack 5.1.2, Python 3.8.10) | Edge device | ONNX/TensorRT, full-app replay, latency/FPS/memory/temperature/power, sustained run. All edge claims come from here. |

Flow: laptop -> GitHub -> Kaggle notebooks -> artifacts (weights, configs, manifests) -> Orin.

## Rules that keep the three machines compatible

1. **Code that runs on the Orin (`src/`, `scripts/`) must run on Python 3.8**, even though the laptop venv is 3.10.
   - No `match` statements, no `X | Y` type unions, no `list[int]` / `dict[str, int]` annotations unless the file starts with `from __future__ import annotations`.
   - Prefer `typing.List`, `typing.Optional`, etc.
2. Use `pathlib` for paths; never hard-code Windows or Linux paths. Paths and tunables live in `configs/*.yaml`.
3. The dataset size is a config value (`subset_fraction`), not a separate copy of the code.
4. Keep the same `ultralytics` version (`8.3.40`) on laptop, Kaggle and Orin so weights load everywhere. Not yet verified on the Orin. YOLO26 may need a newer version; see the plan (open item 9).
5. TensorRT engines are built **on the Orin** from the `.pt`/ONNX weights. Never copy engines between machines.
6. Never fabricate results, annotations or telemetry. Use `NOT MEASURED`, `NOT RUN`, `PENDING` or `BLOCKED` with a reason.

## Layout (adapted from spec section 30)

```
code/
  configs/      # dataset, train_yolo11n/s, adaptive_controller, occupancy, temporal, drift, edge_jetson (.yaml)
  notebooks/    # Kaggle notebooks 00-15
  data/         # manifests/, annotations/ (large data not committed)
  models/       # checkpoints/, onnx/, tensorrt/ (weights and engines not committed)
  src/          # data, detection, occupancy, scene_analysis, slot_risk, controller, temporal,
                #   drift, metrics, edge_monitor, api, reporting
  frontend/     # dashboard
  scripts/      # train, evaluate, run_replay, export_onnx, validate_onnx, build_tensorrt,
                #   benchmark_jetson, collect_jetson_telemetry, check_jetson_environment, generate_*report
  tests/        # unit, integration, data, system, edge, ui
  experiments/  # manifests, logs, metrics, predictions, plots, reports
  docs/         # architecture, implementation_audit, dataset_audit, experiment_protocol,
                #   jetson_deployment, final_implementation_log
  results/      # small CSV/JSON results (committed)
```

These folders are created as work reaches them. Nothing under `data/` or `models/` except manifests and small configs goes into git.

## Laptop environment

Python 3.10 virtual environment in `.venv/`:

```bash
python -m venv .venv            # run with a Python 3.10 interpreter
.venv\Scripts\activate          # Windows
pip install -r requirements-laptop.txt
```

Orin setup: see `ORIN_SETUP.md`.
