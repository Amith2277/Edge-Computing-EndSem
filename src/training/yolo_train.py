"""Helpers for training YOLO11n/s on the notebook 03 export and recording spec section 7 fields.

Pure functions (no ultralytics import) so they are unit-testable and Python 3.8 compatible.
Anything that cannot be measured is recorded as "NOT MEASURED", never guessed.
"""
import csv
import glob
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

NOT_MEASURED = "NOT MEASURED"
REQUIRED = ("model", "imgsz", "epochs", "batch", "optimizer", "patience", "seed")
# Keys passed straight to ultralytics model.train(); everything else in the config is ours.
TRAIN_KEYS = ("imgsz", "epochs", "batch", "optimizer", "patience", "seed", "deterministic", "workers",
              "cache", "save_period", "amp", "fraction")
AUG_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
            "fliplr", "bgr", "mosaic", "mixup", "copy_paste", "copy_paste_mode", "auto_augment", "erasing",
            "crop_fraction", "close_mosaic")
LR_KEYS = ("lr0", "lrf", "cos_lr", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr")


def find_export_dir(search_root, subset_fraction) -> Path:
    """Locate the notebook 03 YOLO export (folder with data.yaml + images/train) under an input root."""
    want = "yolo_f{}".format(subset_fraction)
    cands = sorted(Path(p).parent for p in glob.glob(os.path.join(str(search_root), "**", "data.yaml"),
                                                       recursive=True))
    cands = [c for c in cands if (c / "images" / "train").is_dir()]
    exact = [c for c in cands if c.name == want]
    if exact:
        return exact[0]
    raise FileNotFoundError("No YOLO export named {} (with data.yaml and images/train) under {}. Found: {}. "
                            "Attach notebook 03's output (Save Version first).".format(
                                want, search_root, [str(c) for c in cands] or "none"))


def write_local_data_yaml(export_dir, out_path) -> Path:
    """Write a data.yaml pointing at the export's current location (attached inputs move to /kaggle/input)."""
    export_dir = Path(str(export_dir))
    for s in ("train", "val", "test"):
        if not (export_dir / "images" / s).is_dir():
            raise FileNotFoundError("missing {}".format(export_dir / "images" / s))
    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: car\n".format(
        export_dir.as_posix()))
    return out


def resolve_train_args(cfg: Dict[str, Any], data_yaml, project, name, smoke: bool = False) -> Dict[str, Any]:
    """Build model.train() kwargs from the config. Fails if a required value is still null."""
    missing = [k for k in REQUIRED if cfg.get(k) is None]
    if missing:
        raise ValueError("training config has null required values: {}".format(missing))
    args = {k: cfg[k] for k in TRAIN_KEYS if cfg.get(k) is not None}
    if smoke:
        args.update({k: v for k, v in (cfg.get("smoke") or {}).items() if k in TRAIN_KEYS})
    args.update({"data": str(data_yaml), "project": str(project), "name": str(name), "exist_ok": True,
                 "single_cls": True, "plots": True, "val": True})
    return args


def read_results(results_csv) -> List[Dict[str, float]]:
    """ultralytics results.csv -> list of rows with stripped keys and float values."""
    rows = []
    with open(str(results_csv), newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k.strip()] = float(v)
                except (TypeError, ValueError):
                    row[k.strip()] = v
            rows.append(row)
    return rows


def fitness(row: Dict[str, float]) -> float:
    """Same definition ultralytics uses to pick best.pt."""
    return 0.1 * row.get("metrics/mAP50(B)", 0.0) + 0.9 * row.get("metrics/mAP50-95(B)", 0.0)


def best_epoch(rows: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    b = max(rows, key=fitness)
    keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    return {"epoch": int(b["epoch"]), "fitness": round(fitness(b), 6),
            "metrics": {k: b[k] for k in keys if k in b}}


def latency_summary(ms: List[float]) -> Dict[str, Any]:
    if not ms:
        return {"n": 0, "status": NOT_MEASURED}
    s = sorted(ms)

    def pct(p):
        i = (len(s) - 1) * p / 100.0
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {"n": len(s), "mean_ms": round(sum(s) / len(s), 3), "min_ms": round(s[0], 3),
            "p50_ms": round(pct(50), 3), "p90_ms": round(pct(90), 3), "p95_ms": round(pct(95), 3),
            "p99_ms": round(pct(99), 3), "max_ms": round(s[-1], 3)}


def pick(d: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: d.get(k, NOT_MEASURED) for k in keys}


def build_record(variant: str, train_args: Dict[str, Any], used_args: Dict[str, Any], rows: List[Dict[str, float]],
                 weights_path, weights_sha256: str, split_info: Optional[Dict[str, Any]],
                 duration_s: Optional[float], optimizer_name: Optional[str], model_stats: Dict[str, Any],
                 benchmark: Dict[str, Any], run_id: str, cfg_hash: str, smoke: bool) -> Dict[str, Any]:
    """Assemble the spec section 7 record. `used_args` = the run's args.yaml (what ultralytics actually used)."""
    weights_path = Path(str(weights_path))
    completed = len(rows)
    configured = int(used_args.get("epochs", train_args.get("epochs", 0)) or 0)
    be = best_epoch(rows)
    patience = used_args.get("patience", train_args.get("patience"))
    return {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "smoke_test": bool(smoke),
        "model_variant": variant,
        "status": "fine-tuned from COCO-pretrained {}.pt on MetaPKLot (single class car)".format(variant),
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256": weights_sha256,
        "checkpoint_size_bytes": weights_path.stat().st_size if weights_path.exists() else NOT_MEASURED,
        "dataset_split_hash": (split_info or {}).get("manifest_hash", NOT_MEASURED),
        "subset_fraction": (split_info or {}).get("subset_fraction", NOT_MEASURED),
        "class_names": {0: "car"},
        "imgsz": used_args.get("imgsz", train_args.get("imgsz")),
        "batch": used_args.get("batch", train_args.get("batch")),
        "epochs_configured": configured,
        "epochs_completed": completed,
        "optimizer_config": used_args.get("optimizer", train_args.get("optimizer")),
        "optimizer_used": optimizer_name or NOT_MEASURED,
        "lr_schedule": pick(used_args, LR_KEYS),
        "augmentation": pick(used_args, AUG_KEYS),
        "seed": used_args.get("seed", train_args.get("seed")),
        "deterministic": used_args.get("deterministic", NOT_MEASURED),
        "early_stopping": {"patience": patience,
                           "stopped_early": bool(completed and completed < configured)},
        "best_epoch": be,
        "training_duration_s": round(duration_s, 1) if duration_s is not None else NOT_MEASURED,
        "training_precision": "AMP (mixed fp16/fp32)" if used_args.get("amp", True) else "fp32",
        "parameters": model_stats.get("parameters", NOT_MEASURED),
        "gflops_at_imgsz": model_stats.get("gflops", NOT_MEASURED),
        "kaggle_benchmark": benchmark,
    }
