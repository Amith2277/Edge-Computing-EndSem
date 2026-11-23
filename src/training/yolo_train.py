"""Helpers for training YOLO11n/s on the notebook 03 export and recording spec section 7 fields.

Mostly pure functions (ultralytics is imported only inside disable_integrations) so they are unit-testable and Python 3.8 compatible.
Anything that cannot be measured is recorded as "NOT MEASURED", never guessed.
"""
import csv
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


SKIP_DIRS = {"images", "labels"}  # never descend into YOLO image/label folders (thousands of files)


def find_files(search_root, name: str) -> List[Path]:
    """All files called `name` under search_root. Follows symlinks (Kaggle may mount inputs as links)."""
    out = []
    for cur, dirs, files in os.walk(str(search_root), followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        if name in files:
            out.append(Path(cur) / name)
    return out


def input_tree(search_root, depth: int = 4, max_entries: int = 60) -> str:
    """Short listing of the folders under search_root, to show what is actually attached."""
    root = Path(str(search_root))
    lines = []
    for cur, dirs, files in os.walk(str(root), followlinks=True):
        rel = Path(cur).relative_to(root)
        level = len(rel.parts)
        dirs[:] = sorted(dirs) if level < depth else []
        lines.append("{}{}/  ({} files)".format("  " * level, rel.as_posix() if level else str(root), len(files)))
        if len(lines) >= max_entries:
            lines.append("  ...")
            break
    return "\n".join(lines) if lines else "{} does not exist or is empty".format(root)


def find_export_dir(search_root, subset_fraction) -> Path:
    """Locate the notebook 03 YOLO export (folder with data.yaml + images/train) under an input root."""
    want = "yolo_f{}".format(subset_fraction)
    cands = [p.parent for p in find_files(search_root, "data.yaml")]
    cands = [c for c in cands if (c / "images" / "train").is_dir()]
    exact = [c for c in cands if c.name == want]
    if exact:
        return exact[0]
    raise FileNotFoundError(
        "No YOLO export named {} (with data.yaml and images/train) under {}. Exports found: {}.\n"
        "Attach notebook 03's saved output (Add Input -> Your Work -> notebook 03), or set EXPORT_DIR.\n"
        "What is attached now:\n{}".format(want, search_root, [str(c) for c in cands] or "none",
                                           input_tree(search_root)))


def write_local_data_yaml(export_dir, out_path, splits=("train", "val", "test")) -> Path:
    """Write a data.yaml pointing at the export's current location (attached inputs move to /kaggle/input)."""
    export_dir = Path(str(export_dir))
    for s in splits:
        if not (export_dir / "images" / s).is_dir():
            raise FileNotFoundError("missing {}".format(export_dir / "images" / s))
    out = Path(str(out_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("path: {}\n{}names:\n  0: car\n".format(
        export_dir.as_posix(), "".join("{0}: images/{0}\n".format(s) for s in splits)))
    return out


def _copy_if_missing(pair):
    import shutil
    src, dst = pair
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return 0
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(str(src), str(tmp))
    tmp.replace(dst)
    return 1


def stage_export(export_dir, dest, splits=("train", "val"), workers: int = 16) -> Dict[str, int]:
    """Copy the export's images+labels for `splits` to local disk.

    Kaggle input mounts are read-only and slow (~14 images/s measured), so every epoch's val pass and the
    label scan were I/O bound. Local copies are read fast and let ultralytics save its label cache.
    Resumable: files already copied with the same size are skipped.
    """
    from multiprocessing.pool import ThreadPool
    export_dir, dest = Path(str(export_dir)), Path(str(dest))
    pairs = []
    for s in splits:
        for kind in ("images", "labels"):
            sd = export_dir / kind / s
            if not sd.is_dir():
                raise FileNotFoundError("missing {}".format(sd))
            dd = dest / kind / s
            dd.mkdir(parents=True, exist_ok=True)
            pairs.extend((f, dd / f.name) for f in sd.iterdir() if f.is_file())
    with ThreadPool(workers) as pool:
        copied = sum(pool.imap_unordered(_copy_if_missing, pairs, chunksize=16))
    return {"files": len(pairs), "copied": copied, "skipped_existing": len(pairs) - copied}


INTEGRATIONS = ("raytune", "wandb", "mlflow", "comet", "clearml", "dvc", "neptune", "hub", "tensorboard")


def disable_integrations() -> Dict[str, Any]:
    """Turn off ultralytics experiment-tracker callbacks (Kaggle's preinstalled ray breaks the raytune one,
    wandb can wait for a login). Must run before the trainer is created, in a fresh kernel."""
    from ultralytics import settings
    settings.update({k: False for k in INTEGRATIONS if k in settings})
    return {k: settings[k] for k in INTEGRATIONS if k in settings}


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
