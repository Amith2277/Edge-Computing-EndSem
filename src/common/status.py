"""Progress/status JSON written by every notebook and script (spec section 29)."""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

STATES = ("NOT_RUN", "RUNNING", "COMPLETE", "PARTIAL", "BLOCKED", "FAILED")


def new_run_id(prefix: str = "run") -> str:
    return "{}-{}-{}".format(prefix, time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:6])


def write_status(path, notebook: str, run_id: str, state: str, cfg_hash: Optional[str] = None,
                 message: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if state not in STATES:
        raise ValueError("state must be one of {}".format(STATES))
    payload = {
        "notebook": notebook,
        "run_id": run_id,
        "state": state,
        "config_hash": cfg_hash,
        "message": message,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload["extra"] = extra
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(str(tmp), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)
    return payload


def require_inputs(paths) -> None:
    """Fail early with an actionable message when an input artifact is missing."""
    missing = [str(p) for p in paths if not Path(str(p)).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input artifact(s): {}. Attach the output of the previous notebook "
            "as a Kaggle dataset, or re-run that notebook.".format(", ".join(missing)))
