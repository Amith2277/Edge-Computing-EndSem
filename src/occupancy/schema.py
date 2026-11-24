"""Per-slot output record (spec section 9). One record per slot per processed or skipped frame.

Fields that a later stage owns are filled with NOT_MEASURED until that stage exists (risk: notebook 09,
controller mode: notebook 10), never with a made-up number.
"""
from typing import Any, Dict, List, Optional

NOT_MEASURED = "NOT MEASURED"
STATES = ("occupied", "vacant", "unknown")
FIELDS = ("slot_id", "camera", "frame_id", "timestamp", "state", "score", "score_type", "uncertainty",
          "supporting_detections", "last_reliable_frame", "last_reliable_timestamp", "observation_age_frames",
          "observation_age_seconds", "transition_risk", "controller_mode", "inference", "reason")


def slot_record(slot_id: str, camera: str, frame_id: Any, timestamp: Optional[str], state: str,
                score: Optional[float], uncertainty: Optional[float], support: List[Dict[str, Any]],
                last_reliable_frame: Any, last_reliable_timestamp: Optional[str], age_frames: Optional[int],
                age_seconds: Optional[float], reason: str, inference: str = "executed",
                controller_mode: str = NOT_MEASURED, transition_risk: Any = NOT_MEASURED,
                score_type: str = "uncalibrated detector score (max confidence of supporting detections)") -> Dict[str, Any]:
    if state not in STATES:
        raise ValueError("state must be one of {}".format(STATES))
    if inference not in ("executed", "skipped"):
        raise ValueError("inference must be 'executed' or 'skipped'")
    if inference == "skipped" and state == "vacant" and age_frames == 0:
        raise ValueError("a skipped frame cannot produce a fresh 'vacant' observation")
    rec = {"slot_id": slot_id, "camera": camera, "frame_id": frame_id, "timestamp": timestamp, "state": state,
           "score": score, "score_type": score_type, "uncertainty": uncertainty, "supporting_detections": support,
           "last_reliable_frame": last_reliable_frame, "last_reliable_timestamp": last_reliable_timestamp,
           "observation_age_frames": age_frames, "observation_age_seconds": age_seconds,
           "transition_risk": transition_risk, "controller_mode": controller_mode, "inference": inference,
           "reason": reason}
    assert tuple(rec) == FIELDS
    return rec


def decision_reason(state: str, score: float, conf_thr: float, method: str, n_support: int, roi_frac: float,
                    min_roi: float) -> str:
    if state == "unknown":
        return "only {:.0%} of the slot is inside the observed region (< {:.0%})".format(roi_frac, min_roi)
    if state == "occupied":
        return "{} supporting detection(s) by {}; best score {:.3f} >= threshold {:.3f}".format(
            n_support, method, score, conf_thr)
    return "no detection supports the slot by {} with score >= {:.3f} (best {:.3f})".format(method, conf_thr, score)
