"""Final results (notebook 15): find every saved record, check provenance, read metrics, write tables and the log.

Nothing here computes a new result. Every number comes from a saved record of notebooks 03-14 (and the Orin report
once it exists); a metric without a source is reported as NOT MEASURED with its reason, never estimated.

Paths into records use "/" between keys, and list selectors like `model_latency[model=yolo11n,batch=1]` that pick the
first list element whose fields match (compared as strings).
Python 3.8, standard library + numpy.
"""
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

MISSING = "MISSING"
MODELS = ("yolo11n", "yolo11s")


# ----------------------------------------------------------------------------------------------- record discovery
def sources(frac: Any) -> List[Dict[str, Any]]:
    """What notebooks 03-14 (and the Orin) save, where, and whether the report needs it."""
    s = [{"key": "split", "notebook": "03", "file": "split_info.json", "folder": "splits", "required": True}]
    for v, nb in (("yolo11n", "04"), ("yolo11s", "05")):
        s.append({"key": "train:" + v, "notebook": nb, "file": "training_record.json", "folder": "{}_f{}".format(v, frac), "required": True})
    for v in MODELS:
        s += [{"key": "det:" + v, "notebook": "06", "file": "eval_record.json", "folder": v, "parent": "eval_det", "required": True},
              {"key": "occ:" + v, "notebook": "07", "file": "occupancy_record_{}.json".format(v), "folder": "occupancy", "required": True},
              {"key": "tmp:" + v, "notebook": "08", "file": "temporal_record_{}.json".format(v), "folder": "temporal", "required": True}]
    s += [{"key": "scene", "notebook": "09", "file": "scene_risk_record.json", "folder": "scene_risk", "required": True},
          {"key": "ctl", "notebook": "10", "file": "controller_record.json", "folder": "controller", "required": True},
          {"key": "cost", "notebook": "11", "file": "cost_sensitive_record.json", "folder": "cost_sensitive", "required": True},
          {"key": "drift", "notebook": "12", "file": "drift_record.json", "folder": "drift", "required": True},
          {"key": "eval", "notebook": "13", "file": "evaluation_record.json", "folder": "evaluation", "required": True},
          {"key": "bench", "notebook": "14", "file": "benchmark_record.json", "folder": "benchmark_kaggle", "required": True},
          {"key": "jetson", "notebook": "Orin (plan G8)", "file": "jetson_report.json", "folder": "jetson", "required": False}]
    return s


def discover(root, frac) -> Dict[str, Dict[str, Any]]:
    """{key: {spec, path, record, status, candidates}}; record None when not found."""
    root = Path(root)
    out = {}
    for sp in sources(frac):
        cands = sorted(p for p in root.rglob(sp["file"]) if p.parent.name == sp["folder"]
                       and (sp.get("parent") is None or p.parent.parent.name == sp["parent"]))
        rec, status, path = None, None, None
        if cands:
            path = cands[-1]
            rec = json.loads(path.read_text())
            for st in (path.parent / "status.json", path.parent.parent / "status.json"):
                if st.exists():
                    status = json.loads(st.read_text()); break
        out[sp["key"]] = {"spec": sp, "path": str(path) if path else None, "record": rec, "status": status,
                          "candidates": [str(c) for c in cands]}
    return out


# ----------------------------------------------------------------------------------------------- path access
_SEL = re.compile(r"^(?P<key>[^\[]*)\[(?P<sel>[^\]]*)\]$")


def get(rec: Any, path: str) -> Any:
    """Value at `path` in a record, or MISSING."""
    cur = rec
    if cur is None:
        return MISSING
    for part in path.split("/"):
        m = _SEL.match(part)
        key, sel = (m.group("key"), m.group("sel")) if m else (part, None)
        if key:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return MISSING
        if sel is not None:
            conds = [c.split("=", 1) for c in sel.split(",") if c]
            if not isinstance(cur, list):
                return MISSING
            hit = [x for x in cur if isinstance(x, dict) and all(str(x.get(k)) == v for k, v in conds)]
            if not hit:
                return MISSING
            cur = hit[0]
    return cur


# ----------------------------------------------------------------------------------------------- provenance checks
def checks(found: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []

    def add(name, ok, detail=""):
        rows.append({"check": name, "result": "SKIP" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})

    R = {k: v["record"] for k, v in found.items()}
    for k, v in found.items():
        sp = v["spec"]
        if v["record"] is None:
            add("{} present (notebook {})".format(k, sp["notebook"]), False if sp["required"] else None,
                "not attached" if sp["required"] else "optional, not available")
            continue
        add("{} present (notebook {})".format(k, sp["notebook"]), True, v["path"])
        if len(v["candidates"]) > 1:
            add("{} unique".format(k), False, "{} candidates; using {}".format(len(v["candidates"]), v["path"]))
        if "smoke_test" in v["record"]:
            add("{} is a full run".format(k), not v["record"]["smoke_test"])
        if v["status"] is not None:
            add("{} status COMPLETE".format(k), v["status"].get("state") == "COMPLETE", str(v["status"].get("run_id")))
    h = get(R.get("split"), "manifest_hash")
    for k, r in R.items():
        if r is not None and k != "split" and "dataset_split_hash" in r:
            add("{} split hash = notebook 03".format(k), r["dataset_split_hash"] == h, "{} vs {}".format(r["dataset_split_hash"], h))
    for v in MODELS:
        sha = get(R.get("train:" + v), "checkpoint_sha256")
        for k in ("det:" + v, "occ:" + v):
            if R.get(k) is not None and sha != MISSING:
                add("{} weights = notebook {} checkpoint".format(k, "04" if v == "yolo11n" else "05"), R[k].get("checkpoint_sha256") == sha)
        if R.get("bench") is not None and sha != MISSING:
            add("bench weights {} = training checkpoint".format(v), get(R["bench"], "weights_sha256/" + v) == sha)
    occ_run = get(R.get("occ:yolo11n"), "run_id")
    chain = [("tmp:yolo11n", "notebook07_run_id", occ_run), ("tmp:yolo11s", "notebook07_run_id", occ_run),
             ("scene", "notebook07_run_id", occ_run), ("ctl", "notebook07_run_id", occ_run), ("cost", "notebook07_run_id", occ_run),
             ("drift", "notebook07_run_id", occ_run), ("eval", "notebook07_run_id", occ_run),
             ("ctl", "notebook09_run_id", get(R.get("scene"), "run_id")), ("drift", "notebook09_run_id", get(R.get("scene"), "run_id")),
             ("eval", "notebook09_run_id", get(R.get("scene"), "run_id")), ("eval", "notebook10_run_id", get(R.get("ctl"), "run_id")),
             ("eval", "notebook11_run_id", get(R.get("cost"), "run_id"))]
    for k, field, want in chain:
        if R.get(k) is not None and want != MISSING:
            add("{} {} = upstream run".format(k, field), R[k].get(field) == want, "{} vs {}".format(R[k].get(field), want))
    return rows


# ----------------------------------------------------------------------------------------------- metric catalogue
def _e(section, metric, definition, unit, source=None, path=None, aggregation="", split="test", hardware="",
       per_model=False, fn=None, status=None, reason=None):
    return {"section": section, "metric": metric, "definition": definition, "unit": unit, "source": source, "path": path,
            "aggregation": aggregation, "split": split, "hardware": hardware, "per_model": per_model, "fn": fn,
            "status": status, "reason": reason}


NO_JETSON = "Jetson AGX Orin runbook (plan G) not run yet: no Orin measurement exists"
KAGGLE = "Kaggle (see notebook 14 label)"


def catalogue() -> List[Dict[str, Any]]:
    E, D = _e, "det:{v}"
    c = [
        E("17 detection", "precision", "TP / (TP + FP) at the val-chosen operating confidence", "ratio", D, "test/at_operating_conf/precision", "all test boxes", per_model=True),
        E("17 detection", "recall", "TP / (TP + FN) at the operating confidence", "ratio", D, "test/at_operating_conf/recall", "all test boxes", per_model=True),
        E("17 detection", "F1", "harmonic mean of precision and recall at the operating confidence", "ratio", D, "test/at_operating_conf/f1", per_model=True),
        E("17 detection", "operating confidence", "detector confidence with max val F1", "score", D, "operating_conf/conf", split="val", per_model=True),
        E("17 detection", "AP@0.5", "area under the PR curve at IoU 0.5 (single class: = mAP@0.5)", "ratio", D, "test/mAP50", "ultralytics val", per_model=True),
        E("17 detection", "AP@0.5:0.95", "AP averaged over IoU 0.5..0.95", "ratio", D, "test/mAP50-95", "ultralytics val", per_model=True),
        E("17 detection", "per-class AP", "AP@0.5:0.95 per class (one class: car)", "ratio", D, "test/per_class_ap", per_model=True),
        E("17 detection", "true positives", "matched boxes at the operating confidence (IoU 0.45)", "count", D, "confusion_matrix_at_operating_conf/true_positives", per_model=True),
        E("17 detection", "false positives", "unmatched predictions", "count", D, "confusion_matrix_at_operating_conf/false_positives", per_model=True),
        E("17 detection", "false negatives", "unmatched ground-truth boxes", "count", D, "confusion_matrix_at_operating_conf/false_negatives", per_model=True),
        E("17 detection", "test images", "images in the test split", "count", D, "test/images", per_model=True),
        E("17 detection", "test cars", "ground-truth boxes in the test split", "count", D, "test/cars", per_model=True),
        E("17 detection", "PR curve, F1-confidence curve, confusion matrix plot", "plots per model", "plot", status="PLOT",
          reason="saved by notebook 06 (artifacts/eval_det/<model>/)"),
        E("17 detection", "detection count error", "|predicted - true| cars per image", "count", status="NOT MEASURED",
          reason="not computed; the free-slot count error per image is reported under occupancy"),
        E("17 detection", "confidence distribution, failure examples", "", "", status="NOT MEASURED", reason="not produced by notebook 06"),
        E("18 occupancy", "accuracy", "correct slot decisions / all valid slot decisions", "ratio", "occ:{v}", "test/accuracy", "slot-frames", per_model=True),
        E("18 occupancy", "occupied precision", "", "ratio", "occ:{v}", "test/occupied_precision", per_model=True),
        E("18 occupancy", "occupied recall", "", "ratio", "occ:{v}", "test/occupied_recall", per_model=True),
        E("18 occupancy", "occupied F1", "", "ratio", "occ:{v}", "test/occupied_f1", per_model=True),
        E("18 occupancy", "vacant precision", "", "ratio", "occ:{v}", "test/vacant_precision", per_model=True),
        E("18 occupancy", "vacant recall", "", "ratio", "occ:{v}", "test/vacant_recall", per_model=True),
        E("18 occupancy", "vacant F1", "", "ratio", "occ:{v}", "test/vacant_f1", per_model=True),
        E("18 occupancy", "macro-F1", "mean of occupied and vacant F1", "ratio", "occ:{v}", "test/macro_f1", per_model=True),
        E("18 occupancy", "balanced accuracy", "mean of occupied and vacant recall", "ratio", "occ:{v}", "test/balanced_accuracy", per_model=True),
        E("18 occupancy", "false vacant (count)", "occupied slot shown vacant (nb07 score threshold)", "count", "occ:{v}", "test/FV", per_model=True),
        E("18 occupancy", "false vacant rate", "FV / occupied slots", "ratio", "occ:{v}", "test/false_vacant_rate", per_model=True),
        E("18 occupancy", "false occupied (count)", "vacant slot shown occupied", "count", "occ:{v}", "test/FO", per_model=True),
        E("18 occupancy", "false occupied rate", "FO / vacant slots", "ratio", "occ:{v}", "test/false_occupied_rate", per_model=True),
        E("18 occupancy", "slot decisions", "valid test slot-frames", "count", "occ:{v}", "test/slots", per_model=True),
        E("18 occupancy", "free-slot count MAE", "mean |predicted - true| free slots per image", "slots", "occ:{v}", "test_free_slot_count_error/mae", "per image", per_model=True),
        E("18 occupancy", "unknown slots", "slots < 50% inside the annotation rectangle (never decided)", "count", "occ:{v}", "test_unknown_slots", per_model=True),
        E("18 occupancy", "cost-weighted error (deployed rule)", "(5 FV + FO) per 1000 slot decisions, isotonic P > 1/6", "per 1000", "cost",
          "test_rows[model={v},ratio_fv_fo=5,rule=calibrated isotonic + Bayes]/cost_per_1000", per_model=True),
        E("18 occupancy", "prediction coverage", "share of slot decisions with a probability", "ratio", "cost",
          "test_rows[model={v},ratio_fv_fo=5,rule=calibrated isotonic + Bayes]/coverage", per_model=True),
        E("18 occupancy", "Brier score (isotonic)", "mean (P - y)^2", "", "cost", "calibration/{v}/quality/isotonic/test/brier", per_model=True),
        E("18 occupancy", "ECE (isotonic)", "expected calibration error, 10 bins", "", "cost", "calibration/{v}/quality/isotonic/test/ece", per_model=True),
        E("18 occupancy", "ECE (raw score)", "ECE of the uncalibrated score read as a probability", "", "cost", "calibration/{v}/quality/raw score (not a probability)/test/ece", per_model=True),
        E("18 occupancy", "stale-state fraction (AdaptivePark)", "slot-frames served from an earlier frame", "ratio", "eval",
          "results/test/AdaptivePark (accuracy point) | deployed/stale_share"),
        E("18 occupancy", "per-camera / per-site / per-climate metrics", "", "", status="TABLE", reason="tables T3 (notebook 07 groups) and T6b (notebook 13 sites)"),
        E("18 occupancy", "per-slot metrics", "", "", status="NOT MEASURED", reason="saved per-slot scores exist (notebook 07) but no per-slot table was produced"),
        E("18 occupancy", "reliability diagram", "", "plot", status="PLOT", reason="notebook 11 reliability.png"),
        E("19 temporal", "transition recall", "true state changes shown within 3 frames / all true changes", "ratio", "tmp:{v}", "test_chosen/detection_rate", "nb08 filter", per_model=True),
        E("19 temporal", "missed transitions", "", "count", "tmp:{v}", "test_chosen/missed", per_model=True),
        E("19 temporal", "transition precision / F1", "", "", status="NOT MEASURED",
          reason="notebook 08 saves false transitions and recall but not the predicted-transition count; false transitions per 1000 slot-frames is reported instead"),
        E("19 temporal", "false transitions per 1000 slot-frames", "", "per 1000", "tmp:{v}", "test_chosen/false_per_1000_slot_frames", per_model=True),
        E("19 temporal", "state flips per slot-hour", "predicted state changes per slot-hour", "1/h", "tmp:{v}", "test_chosen/pred_transitions_per_slot_hour", per_model=True),
        E("19 temporal", "mean transition delay", "frames until the output shows a true change", "frames", "tmp:{v}", "test_chosen/mean_delay_frames", per_model=True),
        E("19 temporal", "mean transition delay (minutes)", "", "min", "tmp:{v}", "test_chosen/mean_delay_minutes", per_model=True),
        E("19 temporal", "P90 transition delay", "(P95 and median not saved; P90 is)", "frames", "tmp:{v}", "test_chosen/p90_delay_frames", per_model=True),
        E("19 temporal", "deployed filter: false flips per 1000 (AdaptivePark)", "", "per 1000", "eval", "results/test/AdaptivePark (accuracy point) | deployed/false_flips_per_1000"),
        E("19 temporal", "deployed filter: transition delay (AdaptivePark)", "", "frames", "eval", "results/test/AdaptivePark (accuracy point) | deployed/transition_delay_frames"),
        E("19 temporal", "mean served state age (AdaptivePark)", "", "min", "eval", "results/test/AdaptivePark (accuracy point) | deployed/mean_served_age_min"),
        E("19 temporal", "observations to confirm a change", "k_occ (to occupied) / k_vac (to vacant) of the deployed filter", "frames", "eval", "decision/deployed/filter"),
        E("19 temporal", "median / P95 delay, time-to-correct-state, stale and unknown duration", "", "", status="NOT MEASURED",
          reason="not saved by notebooks 08 / 13 (mean and P90 delay are)"),
        E("20 controller", "total frames", "", "count", "eval", "results/test/AdaptivePark (accuracy point) | deployed/frames"),
        E("20 controller", "model calls", "YOLO11n / YOLO11s invocations", "count", "eval", "results/test/AdaptivePark (accuracy point) | deployed/calls"),
        E("20 controller", "skipped frames", "", "count", "eval", "results/test/AdaptivePark (accuracy point) | deployed/skipped_frames"),
        E("20 controller", "mode distribution", "share of frames per mode", "ratio", "eval", "results/test/AdaptivePark (accuracy point) | deployed/mode_share"),
        E("20 controller", "escalation / refresh reasons", "count of each trigger", "count", "eval", "results/test/AdaptivePark (accuracy point) | deployed/reasons"),
        E("20 controller", "max-age violations", "slot-frames served older than the max age", "count", "eval", "results/test/AdaptivePark (accuracy point) | deployed/max_age_violations"),
        E("20 controller", "mode switches per hour", "", "1/h", "eval", "results/test/AdaptivePark (accuracy point) | deployed/mode_switches_per_hour"),
        E("20 controller", "compute per frame", "YOLO11n units (YOLO11s = 2.63, Kaggle T4 batched ratio)", "n units", "eval",
          "results/test/AdaptivePark (accuracy point) | deployed/cost_per_frame", hardware=KAGGLE),
        E("20 controller", "efficiency point: skipped share", "", "ratio", "eval", "results/test/AdaptivePark (efficiency point) | deployed/skipped_share"),
        E("20 controller", "controller decision overhead", "mean wall time of the decide stage per frame", "ms", "bench",
          "pipeline/adaptive | live/stage_mean_ms/decide", "569 frames", hardware=KAGGLE),
        E("20 controller", "accuracy-efficiency trade-off", "", "", status="TABLE", reason="tables T6 / T7 / T8 (notebook 13)"),
        E("20 controller", "threshold / weight / max-age sensitivity", "", "", status="TABLE", reason="notebook 13 ablations (T7); notebook 10 val grid"),
        E("20 controller", "energy-efficiency trade-off", "", "", status="NOT MEASURED", reason=NO_JETSON),
        E("21 efficiency", "parameters", "", "count", "train:{v}", "parameters", per_model=True),
        E("21 efficiency", "GFLOPs", "at the training input size", "GFLOPs", "train:{v}", "gflops_at_imgsz", per_model=True),
        E("21 efficiency", "checkpoint size", "best.pt", "bytes", "train:{v}", "checkpoint_size_bytes", per_model=True),
        E("21 efficiency", "input resolution", "", "px", "train:{v}", "imgsz", per_model=True),
        E("21 efficiency", "best epoch", "", "epoch", "train:{v}", "best_epoch/epoch", split="val", per_model=True),
        E("21 efficiency", "training time", "", "s", "train:{v}", "training_duration_s", split="train", hardware="Kaggle", per_model=True),
    ]
    for dev, prec, b, lab in (("cuda:0", "fp16", 1, "GPU FP16 batch 1"), ("cuda:0", "fp32", 1, "GPU FP32 batch 1"),
                              ("cuda:0", "fp16", 16, "GPU FP16 batch 16"), ("cpu", "fp32", 1, "CPU FP32 batch 1")):
        sel = "model_latency[model={{v}},device={},precision={},batch={}]".format(dev, prec, b)
        c += [E("21 efficiency", "latency p50, " + lab, "wall time per image, PyTorch", "ms", "bench", sel + "/per_image_ms/p50", "median", hardware=KAGGLE, per_model=True),
              E("21 efficiency", "latency p95, " + lab, "", "ms", "bench", sel + "/per_image_ms/p95", hardware=KAGGLE, per_model=True),
              E("21 efficiency", "latency p99, " + lab, "", "ms", "bench", sel + "/per_image_ms/p99", hardware=KAGGLE, per_model=True),
              E("21 efficiency", "throughput, " + lab, "images per second", "img/s", "bench", sel + "/fps_images", hardware=KAGGLE, per_model=True)]
    c += [E("21 efficiency", "pre / inference / post-processing, GPU FP16 batch 1", "ultralytics stage medians", "ms", "bench",
            "model_latency[model={v},device=cuda:0,precision=fp16,batch=1]/ultralytics_stage_ms_per_image", hardware=KAGGLE, per_model=True),
          E("21 efficiency", "peak GPU memory, GPU FP16 batch 1", "torch max_memory_allocated", "MiB", "bench",
            "model_latency[model={v},device=cuda:0,precision=fp16,batch=1]/gpu_peak_mem_mib", hardware=KAGGLE, per_model=True),
          E("21 efficiency", "YOLO11s / YOLO11n latency ratio", "median per-image latency ratio", "x", "bench", "latency_ratio_s_over_n", hardware=KAGGLE),
          E("21 efficiency", "model loading time, average memory, max latency, performance per watt", "", "", status="NOT MEASURED",
            reason="not saved by notebook 14; Orin measurements pending (plan G7)")]
    for m in ("preprocessing / inference / postprocessing latency", "slot-association latency", "temporal / controller latency",
              "end-to-end frame latency (mean, P50, P90, P95, P99, max)", "FPS, dropped frames", "CPU / GPU utilisation", "RAM / GPU memory, memory growth",
              "model loading and startup time", "temperature, clocks, throttling events", "average / peak power, energy per frame, performance per watt",
              "uptime, failures, recovery, API errors", "TensorRT FP16 vs PyTorch (n, s)", "INT8"):
        c.append(E("22 Jetson AGX Orin", m, "measured on the Orin", "", source="jetson", status="NOT MEASURED", reason=NO_JETSON))
    c += [E("22 Jetson AGX Orin (Kaggle stand-in)", "end-to-end frame latency, AdaptivePark", "FramePipeline per frame incl. decode", "ms", "bench",
            "pipeline/adaptive | live/total_ms/mean", hardware=KAGGLE + " - NOT Jetson"),
          E("22 Jetson AGX Orin (Kaggle stand-in)", "slot-association latency", "", "ms", "bench", "pipeline/adaptive | live/stage_mean_ms/associate", hardware=KAGGLE + " - NOT Jetson"),
          E("22 Jetson AGX Orin (Kaggle stand-in)", "ONNX export", "status / box match at the operating threshold", "", "bench", "onnx/{v}/status", hardware=KAGGLE, per_model=True),
          E("22 Jetson AGX Orin (Kaggle stand-in)", "ONNX box match at operating threshold", "share of PyTorch boxes matched by ONNX Runtime (IoU 0.5)", "ratio",
            "bench", "onnx/{v}/box_match_share_at_operating_threshold", hardware=KAGGLE, per_model=True)]
    c += [E("16 drift", "drift test chosen on val", "", "", "drift", "test_chosen_on_val", split="val"),
          E("16 drift", "window", "rolling window", "frames", "drift", "window_frames"),
          E("16 drift", "clean val alert rate", "share of clean val windows at warning or critical (target <= 5%)", "ratio", "drift", None, split="val",
            fn=lambda r: None if r.get("thresholds") is None else round(1 - r["thresholds"][r["test_chosen_on_val"]]["clean_rate"]["ok"], 4)),
          E("16 drift", "natural test drift alerts", "windows warning or worse, by site", "ratio", "drift", "natural_test/by_site"),
          E("16 drift", "re-detection agreement", "re-run YOLO11n vs cached notebook 07 slot decisions", "ratio", "drift", "redetection_agreement_with_nb07"),
          E("16 drift", "synthetic drift alert rates", "", "", status="TABLE", reason="table T9 (notebook 12)")]
    return c


def resolve(entry: Dict[str, Any], R: Dict[str, Any], model: Optional[str] = None) -> Any:
    if entry["status"] is not None:
        return entry["status"]
    key = entry["source"].replace("{v}", model or "")
    rec = R.get(key)
    if rec is None:
        return "NOT MEASURED" if key == "jetson" else MISSING
    if entry["fn"] is not None:
        try:
            return entry["fn"](rec)
        except (KeyError, TypeError, ZeroDivisionError):
            return MISSING
    return get(rec, entry["path"].replace("{v}", model or ""))


def catalogue_rows(R: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for e in catalogue():
        vals = {v: resolve(e, R, v) for v in MODELS} if e["per_model"] else {"value": resolve(e, R)}
        rows.append({"section": e["section"], "metric": e["metric"], "definition": e["definition"], "unit": e["unit"],
                     "aggregation": e["aggregation"], "split": e["split"], "hardware": e["hardware"] or ("" if e["status"] else "replay of saved detections / Kaggle"),
                     "source": "" if e["source"] is None else "{} : {}".format(e["source"], e["path"] or "(derived)"),
                     "values": vals, "note": e["reason"] or ""})
    return rows


# ----------------------------------------------------------------------------------------------- markdown
def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return ("{:." + str(nd) + "g}").format(x) if abs(x) < 1e5 else "{:.3g}".format(x)
    if isinstance(x, dict):
        return ", ".join("{} {}".format(k, fmt(v, nd)) for k, v in x.items())
    if isinstance(x, list):
        return ", ".join(fmt(v, nd) for v in x)
    return str(x)


def md_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str], headers: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    h = list(headers or cols)
    out = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out) + "\n"


def overall_status(found: Dict[str, Dict[str, Any]], repo_has: Dict[str, bool], check_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """COMPLETE only with every required record, all checks passing, the Orin report, the API and the dashboard."""
    missing = [k for k, v in found.items() if v["spec"]["required"] and v["record"] is None]
    failed = [r["check"] for r in check_rows if r["result"] == "FAIL"]
    gaps = []
    if found.get("jetson", {}).get("record") is None:
        gaps.append("Jetson AGX Orin deployment and measurements (plan G)")
    for k, ok in repo_has.items():
        if not ok:
            gaps.append(k)
    state = "BLOCKED" if missing else ("COMPLETE" if not failed and not gaps else "PARTIAL")
    return {"state": state, "missing_records": missing, "failed_checks": failed, "gaps": gaps}
