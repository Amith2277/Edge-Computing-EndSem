"""Leakage-aware train/val/test splits for MetaPKLot (spec section 6). Python 3.8 compatible.

Grouping unit is (site, day):
  - Frames are 5 min (PKLot) or 30 min (CNRPark-EXT) apart, so frames of one day are near-duplicates and must
    stay in one split.
  - The nine CNRPark-EXT cameras watch one site on the same days, so a day is assigned for the whole site, not
    per camera (otherwise the same cars/weather would be in train and test through another viewpoint).
  - PKLot lots PUCPR, UFPR04, UFPR05 are treated as separate sites.
Optionally whole cameras can be held out (`holdout_cameras`) for a camera-held-out generalization test.
"""
import hashlib
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SPLITS = ("train", "val", "test")


def site_of(camera: str) -> str:
    """'CNRPark-EXT/camera3' -> 'CNRPark-EXT'; 'PKLot/UFPR04' -> 'UFPR04'."""
    parts = camera.split("/")
    return parts[0] if parts[0] == "CNRPark-EXT" else parts[-1]


def normalize_day(text: str) -> Optional[str]:
    m = re.match(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if not m:
        return None
    return "{:04d}-{:02d}-{:02d}".format(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_day_lists(day_lists: Dict[str, List[str]]) -> Dict[Tuple[str, str], str]:
    """{(site, 'YYYY-MM-DD'): 'reason; reason'} from the dataset's non_working_days / non_representative_days files."""
    out = defaultdict(set)  # type: Dict[Tuple[str, str], Set[str]]
    for path, lines in day_lists.items():
        parts = path.replace("\\", "/").split("/")
        if "CNRPark-EXT" in parts:
            site = "CNRPark-EXT"
        elif "PKLot" in parts and len(parts) > parts.index("PKLot") + 1:
            site = parts[parts.index("PKLot") + 1]
        else:
            continue  # PLds: no images in this mirror
        for line in lines:
            day = normalize_day(line)
            if day:
                reason = line.split("/", 1)[1].strip().lower() if "/" in line else "listed"
                out[(site, day)].add(reason)
    return {k: "; ".join(sorted(v)) for k, v in out.items()}


def _site_days(images: Iterable[Dict[str, Any]], holdout: Set[str]):
    days = defaultdict(set)  # type: Dict[str, Set[str]]
    climates = defaultdict(lambda: defaultdict(set))  # type: Dict[str, Dict[str, Set[str]]]
    for img in images:
        if img["camera"] in holdout:
            continue
        s = site_of(img["camera"])
        days[s].add(img["day"])
        climates[s][img["climate"]].add(img["day"])
    return {s: sorted(d) for s, d in days.items()}, climates


def assign_splits(images: List[Dict[str, Any]], ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 42,
                  max_tries: int = 500, holdout_cameras: Iterable[str] = (),
                  min_days_for_coverage: int = 3) -> Tuple[Dict[Any, str], Dict[str, Any]]:
    """Assign every image to train/val/test (or holdout_test) by (site, day).

    Retries the seeded shuffle until every climate that occurs on >= `min_days_for_coverage` days of a site appears
    in that site's val and test days. Returns ({image_id: split}, info).
    """
    if abs(sum(ratios) - 1.0) > 1e-6 or len(ratios) != 3:
        raise ValueError("ratios must be three values summing to 1")
    holdout = set(holdout_cameras)
    site_days, climates = _site_days(images, holdout)
    chosen, tries, coverage_ok = None, 0, False
    for tries in range(max_tries):
        rng = random.Random(seed + tries)
        plan = {}  # type: Dict[str, Dict[str, str]]
        for site in sorted(site_days):
            days = list(site_days[site])
            rng.shuffle(days)
            n = len(days)
            n_test = max(1, int(round(ratios[2] * n))) if n >= 3 else 0
            n_val = max(1, int(round(ratios[1] * n))) if n >= 3 else 0
            plan[site] = {}
            for i, d in enumerate(days):
                plan[site][d] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        ok = True
        for site, cl in climates.items():
            for climate, cdays in cl.items():
                if len(cdays) < min_days_for_coverage:
                    continue
                got = {plan[site][d] for d in cdays}
                if not {"val", "test"} <= got:
                    ok = False
        chosen = plan
        if ok:
            coverage_ok = True
            break
    assignments = {}
    for img in images:
        if img["camera"] in holdout:
            assignments[img["image_id"]] = "holdout_test"
        else:
            assignments[img["image_id"]] = chosen[site_of(img["camera"])][img["day"]]
    info = {"seed": seed, "seed_used": seed + tries, "tries": tries + 1, "climate_coverage_ok": coverage_ok,
            "ratios": list(ratios), "holdout_cameras": sorted(holdout),
            "days_per_split": {site: {sp: sorted(d for d, s in plan.items() if s == sp) for sp in SPLITS}
                               for site, plan in chosen.items()}}
    return assignments, info


def subsample_train(images: List[Dict[str, Any]], assignments: Dict[Any, str], fraction: float) -> Set[Any]:
    """Image ids kept for training when using a subset: every k-th frame of each (camera, day), k = round(1/f).

    Val/test/holdout images are always kept, so evaluation does not change with the subset size.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(round(1.0 / fraction)))
    keep = set()
    groups = defaultdict(list)  # type: Dict[Tuple[str, str], List[Dict[str, Any]]]
    for img in images:
        if assignments[img["image_id"]] == "train":
            groups[(img["camera"], img["day"])].append(img)
        else:
            keep.add(img["image_id"])
    for _key, imgs in groups.items():
        imgs.sort(key=lambda i: i["timestamp"] or "")
        keep.update(i["image_id"] for i in imgs[::k])
    return keep


def manifest_rows(images: List[Dict[str, Any]], assignments: Dict[Any, str], keep: Set[Any],
                  day_flags: Optional[Dict[Tuple[str, str], str]] = None) -> List[Dict[str, Any]]:
    flags = day_flags or {}
    rows = []
    for img in sorted(images, key=lambda i: i["file_name"]):
        site = site_of(img["camera"])
        rows.append({"image_id": img["image_id"], "file_name": img["file_name"], "camera": img["camera"],
                     "site": site, "day": img["day"], "timestamp": img["timestamp"], "climate": img["climate"],
                     "split": assignments[img["image_id"]], "in_subset": img["image_id"] in keep,
                     "day_flag": flags.get((site, img["day"]), "")})
    return rows


def manifest_hash(rows: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: str(r["image_id"])):
        h.update("{},{},{}\n".format(r["image_id"], r["split"], int(bool(r["in_subset"]))).encode())
    return h.hexdigest()[:16]


def split_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = defaultdict(lambda: defaultdict(int))  # type: Dict[str, Dict[str, int]]
    for r in rows:
        out[r["split"]]["images"] += 1
        out[r["split"]]["in_subset"] += int(bool(r["in_subset"]))
        out[r["split"]]["site:" + r["site"]] += 1
        out[r["split"]]["climate:" + r["climate"]] += 1
        if r["day_flag"]:
            out[r["split"]]["flagged_day_images"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def check_no_leakage(rows: List[Dict[str, Any]]) -> List[str]:
    """Every (site, day) must map to exactly one split (holdout cameras excluded). Returns violations."""
    seen = defaultdict(set)  # type: Dict[Tuple[str, str], Set[str]]
    for r in rows:
        if r["split"] != "holdout_test":
            seen[(r["site"], r["day"])].add(r["split"])
    return ["{} {} -> {}".format(s, d, sorted(v)) for (s, d), v in sorted(seen.items()) if len(v) > 1]
