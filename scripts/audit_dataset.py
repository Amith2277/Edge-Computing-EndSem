"""Run the dataset audit from the command line.

Example:
    python scripts/audit_dataset.py --root /kaggle/input/<slug> --out /kaggle/working/artifacts/audit --smoke
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import config_hash, load_yaml, resolve_data_root  # noqa: E402
from src.common.status import new_run_id, write_status  # noqa: E402
from src.data.audit import run_audit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml"))
    ap.add_argument("--root", default=None, help="dataset root (overrides config/env)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="audit an evenly spaced sample only")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = resolve_data_root(cfg, args.root)
    if root is None:
        print("ERROR: no dataset root. Pass --root, set ADAPTIVEPARK_DATA_ROOT, or fill dataset.root.")
        return 2
    a = cfg.get("audit", {})
    run_id, cfg_h = new_run_id("audit"), config_hash(cfg)
    status = Path(args.out) / "status.json"
    write_status(status, "01_dataset_audit", run_id, "RUNNING", cfg_h)
    try:
        nd = a.get("near_duplicate", {})
        summary = run_audit(root, args.out, smoke=args.smoke, smoke_max_files=a.get("smoke_max_files", 200),
                            compute_hashes=a.get("compute_hashes", True), near_dup=nd.get("enabled", True),
                            hash_size=nd.get("hash_size", 8), max_hamming=nd.get("max_hamming", 2),
                            tree_depth=a.get("tree_depth", 3), image_exts=a.get("image_exts"),
                            ann_exts=a.get("annotation_exts"), video_exts=a.get("video_exts"))
    except Exception as exc:
        write_status(status, "01_dataset_audit", run_id, "FAILED", cfg_h, str(exc))
        raise
    write_status(status, "01_dataset_audit", run_id, "PARTIAL" if args.smoke else "COMPLETE", cfg_h,
                 "smoke sample only" if args.smoke else "")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
