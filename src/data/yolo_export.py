"""Export MetaPKLot images + car boxes to Ultralytics YOLO format. Python 3.8 compatible.

Only objects inside each image's annotationsRectangle are guaranteed to be labelled, so pixels outside the rectangle
are filled with a neutral gray (114, the Ultralytics letterbox value). Otherwise unlabelled cars there would teach the
detector that cars are background. The same region-of-interest mask is applied at inference time (deployment config),
so train and test see the same input.
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

FILL = (114, 114, 114)


def safe_name(file_name: str) -> str:
    """'PKLot/PUCPR/Sunny/2012-09-11/x.jpg' -> 'PKLot__PUCPR__Sunny__2012-09-11__x.jpg' (unique and flat)."""
    return file_name.replace("/", "__")


def rect_covers_image(rect: Optional[Sequence[float]], w: int, h: int) -> bool:
    if not rect:
        return True
    x, y, rw, rh = rect
    return x <= 0 and y <= 0 and x + rw >= w and y + rh >= h


def yolo_label_lines(bboxes: List[Sequence[float]], w: int, h: int, rect: Optional[Sequence[float]] = None,
                     cls: int = 0, min_px: float = 2.0) -> List[str]:
    """COCO [x, y, bw, bh] boxes -> YOLO 'cls cx cy w h' lines (normalized), clipped to the ROI and image."""
    x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
    if rect:
        x0, y0 = max(x0, rect[0]), max(y0, rect[1])
        x1, y1 = min(x1, rect[0] + rect[2]), min(y1, rect[1] + rect[3])
    lines = []
    for b in bboxes:
        bx0, by0 = max(x0, b[0]), max(y0, b[1])
        bx1, by1 = min(x1, b[0] + b[2]), min(y1, b[1] + b[3])
        bw, bh = bx1 - bx0, by1 - by0
        if bw < min_px or bh < min_px:
            continue
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls, (bx0 + bw / 2) / w, (by0 + bh / 2) / h, bw / w, bh / h))
    return lines


def export_image(src: Path, dst: Path, rect: Optional[Sequence[float]], w: int, h: int, quality: int = 95) -> str:
    """Copy as-is when the ROI is the whole image, else write a gray-masked copy. Returns 'copied'/'masked'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rect_covers_image(rect, w, h):
        shutil.copyfile(str(src), str(dst))
        return "copied"
    from PIL import Image
    with Image.open(str(src)) as im:
        im = im.convert("RGB")
        out = Image.new("RGB", im.size, FILL)
        x, y, rw, rh = [int(round(v)) for v in rect]
        box = (max(0, x), max(0, y), min(im.width, x + rw), min(im.height, y + rh))
        out.paste(im.crop(box), box[:2])
        out.save(str(dst), quality=quality)
    return "masked"


def export_dataset(dataset_root, out_dir, rows: List[Dict[str, Any]], images_by_id: Dict[Any, Dict[str, Any]],
                   boxes_by_image: Dict[Any, List[Sequence[float]]], splits: Sequence[str] = ("train", "val", "test"),
                   workers: int = 8) -> Dict[str, Any]:
    """Write images/<split>/*.jpg and labels/<split>/*.txt for manifest rows with in_subset=True. Resumable."""
    root, out = Path(str(dataset_root)), Path(str(out_dir))
    todo = [r for r in rows if r["split"] in splits and r["in_subset"]]
    counts = {"copied": 0, "masked": 0, "skipped_existing": 0, "boxes": 0, "empty_label_images": 0}

    def one(r: Dict[str, Any]) -> Tuple[str, int]:
        img = images_by_id[r["image_id"]]
        name = safe_name(r["file_name"])
        dst_img = out / "images" / r["split"] / name
        dst_lbl = out / "labels" / r["split"] / (os.path.splitext(name)[0] + ".txt")
        lines = yolo_label_lines(boxes_by_image.get(r["image_id"], []), img["width"], img["height"], img["rect"])
        if dst_img.exists() and dst_lbl.exists():
            return "skipped_existing", len(lines)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        return export_image(root / r["file_name"], dst_img, img["rect"], img["width"], img["height"]), len(lines)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for kind, n in ex.map(one, todo):
            counts[kind] += 1
            counts["boxes"] += n
            counts["empty_label_images"] += int(n == 0)
    counts["images"] = len(todo)
    return counts


def write_data_yaml(out_dir, names: Dict[int, str] = None, splits: Sequence[str] = ("train", "val", "test")) -> Path:
    out = Path(str(out_dir))
    names = names or {0: "car"}
    lines = ["path: {}".format(out.as_posix())]
    for s in splits:
        lines.append("{}: images/{}".format(s, s))
    lines.append("names:")
    for k in sorted(names):
        lines.append("  {}: {}".format(k, names[k]))
    p = out / "data.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p
