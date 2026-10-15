"""CPU-only label, reward and data checks; official scorer stays unchanged."""
import json
import math
import random
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import parse_output, _match

PROMPT = (
    "Locate all road damage in this image. Detect only pothole, crack, manhole. "
    "Return only a JSON list, no markdown or explanation. Each object has label and "
    "bbox_2d: [x1,y1,x2,y2], integer coordinates normalized to 0-1000 relative to "
    "image width and height. Sort by top edge then left edge. If none, return []."
)
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def yolo_to_box(label, yolo):
    if label not in CLASSES or len(yolo) != 4:
        raise ValueError("Unknown label or malformed YOLO box")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError("YOLO values must be finite and in [0,1]")
    xc, yc, w, h = yolo
    # The official loader accepts zero dimensions. Preserve these annotations;
    # expand only the serialized training target to one coordinate unit below.
    x1, y1, x2, y2 = [int(round(max(0, min(1, v)) * 1000))
                       for v in (xc-w/2, yc-h/2, xc+w/2, yc+h/2)]
    # Preserve positive area even when a tiny box rounds to the boundary 1000.
    x1, y1 = min(x1, 999), min(y1, 999)
    x2, y2 = max(x2, x1+1), max(y2, y1+1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(o["label"], o["yolo"]) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"], b["bbox_2d"][2:]))
    return json.dumps(boxes, separators=(",", ":"))


def iou_reward(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    _, ious = _match(gt_boxes, pred)
    return 0.2 + 0.8 * sum(ious) / max(len(gt_boxes), len(pred), 1)


def train_subset(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    ordered = sorted(rows, key=lambda r: r["image_id"])
    random.Random(seed).shuffle(ordered)
    return ordered[:max(1, int(len(ordered)*fraction))]


def audit_data():
    rows = {s: read_jsonl(f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
    expected = {"train": (1422, [1074, 1750, 691]), "val": (276, [66, 470, 103]),
                "test": (311, [121, 299, 163])}
    stats, sessions, ids = {}, set(), set()
    zero_area = []
    for s, entries in rows.items():
        group = {e["session"] for e in entries}
        current_ids = {e["image_id"] for e in entries}
        assert not sessions & group, f"Session leakage: {s}"
        assert not ids & current_ids and len(current_ids) == len(entries), "Duplicate image"
        sessions |= group
        ids |= current_ids
        counts = [sum(o["label"] == c for e in entries for o in e["objects"]) for c in CLASSES]
        assert (len(entries), counts) == expected[s], (s, len(entries), counts)
        for e in entries:
            assert (e["width"], e["height"]) == (640, 360)
            assert (Path("data") / e["image_path"]).is_file()
            objects_to_answer(e["objects"])
            for index, obj in enumerate(e["objects"]):
                if obj["yolo"][2] == 0 or obj["yolo"][3] == 0:
                    zero_area.append({"split": s, "image_id": e["image_id"],
                                      "object_index": index, "label": obj["label"],
                                      "original_yolo": obj["yolo"],
                                      "serialized_target": yolo_to_box(obj["label"], obj["yolo"])})
        stats[s] = {"images": len(entries), "sessions": len(group), "boxes": dict(zip(CLASSES, counts))}
    meta = json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))
    assert meta["dataset_version"].startswith("d10-")
    dump("report/diagnostics/data_audit.json", {"dataset_version": meta["dataset_version"], "splits": stats})
    dump("report/diagnostics/zero_area_annotations.json",
         {"count": len(zero_area), "annotations": zero_area,
          "policy": "Keep original splits and scorer GT unchanged; expand serialized target to at least 1/1000 per axis."})
    print(json.dumps(stats, indent=2))
    print(f"Zero-area source annotations: {len(zero_area)} (see report/diagnostics/zero_area_annotations.json)")
    return rows
