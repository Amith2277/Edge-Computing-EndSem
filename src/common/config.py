"""Config loading and hashing. Python 3.8 compatible."""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of a config dict (key order independent)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def find_nulls(cfg: Any, prefix: str = "") -> list:
    """Return dotted paths of keys whose value is None (the NEEDS_TUNING markers)."""
    out = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            p = prefix + "." + str(k) if prefix else str(k)
            if v is None:
                out.append(p)
            else:
                out.extend(find_nulls(v, p))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            p = "{}[{}]".format(prefix, i)
            if v is None:
                out.append(p)
            else:
                out.extend(find_nulls(v, p))
    return out


def resolve_data_root(cfg: Dict[str, Any], override: Optional[str] = None) -> Optional[Path]:
    """Priority: explicit override > env ADAPTIVEPARK_DATA_ROOT > config dataset.root."""
    value = override or os.environ.get("ADAPTIVEPARK_DATA_ROOT") or cfg.get("dataset", {}).get("root")
    return Path(value) if value else None
