"""Parking-slot geometry: a persistent slot registry built from the MetaPKLot spot polygons, plus validators.

Pure Python (no shapely), Python 3.8 compatible, so the same code runs on Kaggle and on the Orin.
Coordinates are pixels of the original image (origin top-left, x right, y down). Notebook 03 exports images at their
original size (it only grays out pixels outside the annotation rectangle), so slot polygons apply to the exported
images unchanged; notebook 07 checks this on a sample.

Spot polygons are fixed per camera (notebook 02), so a persistent slot id is (camera, polygon). Ids are numbered by
polygon centroid (top to bottom, then left to right) inside each camera: "PKLot/PUCPR#007".
"""
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
COORD_SYSTEM = "pixels of the original image; origin top-left; x right, y down"


def to_points(flat: Sequence[float]) -> List[Point]:
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]


def area(pts: Sequence[Point]) -> float:
    """Shoelace area (absolute)."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def centroid(pts: Sequence[Point]) -> Point:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def bounds(pts: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def clip_to_rect(pts: Sequence[Point], x0: float, y0: float, x1: float, y1: float) -> List[Point]:
    """Sutherland-Hodgman: the part of polygon `pts` inside the axis-aligned rectangle (works for any simple subject
    polygon because the clip region is convex)."""
    def clip(poly, inside, cross):
        out = []
        n = len(poly)
        for i in range(n):
            cur, prev = poly[i], poly[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def x_cross(xc):
        return lambda p, q: (xc, p[1] + (q[1] - p[1]) * (xc - p[0]) / (q[0] - p[0]))

    def y_cross(yc):
        return lambda p, q: (p[0] + (q[0] - p[0]) * (yc - p[1]) / (q[1] - p[1]), yc)

    poly = list(pts)
    for inside, cross in ((lambda p: p[0] >= x0, x_cross(x0)), (lambda p: p[0] <= x1, x_cross(x1)),
                          (lambda p: p[1] >= y0, y_cross(y0)), (lambda p: p[1] <= y1, y_cross(y1))):
        if not poly:
            break
        poly = clip(poly, inside, cross)
    return poly


def rect_intersection_area(pts: Sequence[Point], x0: float, y0: float, x1: float, y1: float) -> float:
    return area(clip_to_rect(pts, x0, y0, x1, y1)) if x1 > x0 and y1 > y0 else 0.0


def point_in_polygon(x: float, y: float, pts: Sequence[Point]) -> bool:
    """Ray casting (even-odd)."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 1e-9) - (v < -1e-9)
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0   # proper crossing only (touching endpoints is not a self-intersection)


def is_simple(pts: Sequence[Point]) -> bool:
    """No two non-adjacent edges cross (a bow-tie quadrilateral is invalid geometry)."""
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True


def polygon_key(flat: Sequence[float], ndigits: int = 0) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in flat)


# ----------------------------------------------------------------------------------------------- registry
def build_registry(images: Iterable[Dict[str, Any]], spots: Iterable[Dict[str, Any]],
                   provenance: str = "MetaPKLot annotations/original/spots") -> Tuple[Dict[str, Dict[str, Any]], Dict[Any, str]]:
    """Slots per camera from spot annotations.

    images: rows with image_id, camera, width, height.  spots: rows with ann_id, image_id, polygon (flat list).
    Returns (slots by slot_id, slot_id by ann_id).
    """
    img = {i["image_id"]: i for i in images}
    polys = defaultdict(dict)       # camera -> key -> first flat polygon
    seen = defaultdict(Counter)     # camera -> key -> number of images
    sizes = defaultdict(set)        # camera -> {(w, h)}
    ann_key = {}
    for s in spots:
        im = img.get(s["image_id"])
        if im is None or not s["polygon"]:
            continue
        cam, k = im["camera"], polygon_key(s["polygon"])
        polys[cam].setdefault(k, list(map(float, s["polygon"])))
        seen[cam][k] += 1
        sizes[cam].add((int(im["width"]), int(im["height"])))
        ann_key[s["ann_id"]] = (cam, k)
    slots, id_of = {}, {}
    for cam in sorted(polys):
        w, h = sorted(sizes[cam])[0]
        order = sorted(polys[cam], key=lambda k: (round(centroid(to_points(polys[cam][k]))[1]),
                                                  centroid(to_points(polys[cam][k]))[0]))
        for n, k in enumerate(order):
            sid = "{}#{:03d}".format(cam, n)
            id_of[(cam, k)] = sid
            slots[sid] = {"slot_id": sid, "camera": cam, "polygon": polys[cam][k], "ref_width": w, "ref_height": h,
                          "ref_sizes_seen": sorted(sizes[cam]), "coordinate_system": COORD_SYSTEM,
                          "provenance": provenance, "images_seen": seen[cam][k]}
    return slots, {a: id_of[ck] for a, ck in ann_key.items()}


def validate_slots(slots: Dict[str, Dict[str, Any]], spot_slot_ids: Optional[Dict[Any, Tuple[Any, str]]] = None,
                   tol: float = 1.0) -> Dict[str, Any]:
    """Spec section 9 checks. spot_slot_ids: ann_id -> (image_id, slot_id), used to find a slot twice in one image."""
    bad = {"out_of_range": [], "zero_area": [], "not_simple": [], "too_few_points": [], "mixed_ref_sizes": []}
    for sid, s in slots.items():
        pts = to_points(s["polygon"])
        if len(pts) < 3:
            bad["too_few_points"].append(sid)
            continue
        w, h = s["ref_width"], s["ref_height"]
        if not all(-tol <= x <= w + tol and -tol <= y <= h + tol for x, y in pts):
            bad["out_of_range"].append(sid)
        if area(pts) <= 0:
            bad["zero_area"].append(sid)
        if not is_simple(pts):
            bad["not_simple"].append(sid)
        if len(s.get("ref_sizes_seen", [])) > 1:
            bad["mixed_ref_sizes"].append(sid)
    ids = [s["slot_id"] for s in slots.values()]
    dup_ids = [k for k, c in Counter(ids).items() if c > 1]
    dup_in_image = 0
    if spot_slot_ids:
        c = Counter(spot_slot_ids.values())
        dup_in_image = sum(1 for v in c.values() if v > 1)
    per_cam = Counter(s["camera"] for s in slots.values())
    return {"slots": len(slots), "slots_per_camera": dict(sorted(per_cam.items())),
            "duplicate_slot_ids": dup_ids, "same_slot_twice_in_one_image": dup_in_image,
            **{k: {"count": len(v), "sample": v[:5]} for k, v in bad.items()},
            "ok": not dup_ids and dup_in_image == 0 and not any(bad[k] for k in ("out_of_range", "zero_area",
                                                                                  "not_simple", "too_few_points"))}


def roi_fraction(polygon: Sequence[float], rect: Optional[Sequence[float]]) -> float:
    """Fraction of the slot polygon inside the annotation rectangle [x, y, w, h] (1.0 when there is no rectangle).
    Outside the rectangle the exported image is gray, so a slot mostly outside it cannot be observed."""
    pts = to_points(polygon)
    a = area(pts)
    if not rect or a <= 0:
        return 1.0
    x, y, w, h = rect
    return rect_intersection_area(pts, x, y, x + w, y + h) / a
