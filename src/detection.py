import time
import cv2
from pathlib import Path

from src.model_loader import get_damage_model

# =============================
# CONFIG
# =============================
BASE_CONF = 0.45

# Vertical zones (fraction of image height)
TIRE_MIN_Y = 0.75               # tires sit low in the frame
GLASS_MAX_Y = 0.55              # glass sits high (cockpit / windscreen)

# Area thresholds (fraction of image area)
MIN_BOX_HEIGHT = 0.03
MAX_TIRE_AREA = 0.15
GLASS_MISSING_SINGLE = 0.12     # one big hole
GLASS_MISSING_TOTAL = 0.18      # aggregated damage

# Confidence
MIN_GLASS_CONF = 0.70           # below this a raw "glass_shatter" is downgraded

# Per-class confidence floors, applied to the FINAL (post-reclassification)
# class so no class is silently trusted. Classes with a geometric prior
# (tire/glass) keep the base bar; the geometrically-unconstrained and noisier
# CarDD classes (deformation, other_damage, lamp_broken) are held higher, since
# they are the ones that fire on non-damage such as helmets or liveries.
MIN_CONF_BY_CLASS = {
    "tire_flat": 0.45,
    "glass_shatter": 0.55,
    "glass_missing": 0.55,
    "dent": 0.55,
    "scratch": 0.55,
    "lamp_broken": 0.55,
    "deformation": 0.60,
    "other_damage": 0.60,
}
DEFAULT_MIN_CONF = 0.55


# =============================
# DAMAGE CLASS VALIDATION
# =============================
def validate_damage_class(
    raw_class,
    confidence,
    x1, y1, x2, y2,
    img_h, img_w
):
    """Validate/relabel one raw damage detection; return the accepted class or None.

    Covers all 7 CarDD classes:
      * ``tire_flat`` / ``glass_shatter`` get geometric reclassification (tires
        must sit low; oversized "tires" are mislabeled glass; low-confidence or
        low-placed glass is downgraded to a scratch/missing panel).
      * ``dent``, ``scratch``, ``lamp_broken``, ``deformation`` and
        ``other_damage`` have no geometric prior, so they rely on the per-class
        confidence floor below.
    Every final class is confidence-gated via ``MIN_CONF_BY_CLASS``, so a class
    is never passed through untouched the way ``deformation``/``other_damage``
    previously were.
    """
    box_center_y = (y1 + y2) / 2
    box_height = y2 - y1
    box_area = (x2 - x1) * (y2 - y1)
    img_area = (img_h * img_w) or 1
    area_ratio = box_area / img_area

    # 1) Noise: too-small boxes are dropped regardless of class.
    if box_height < img_h * MIN_BOX_HEIGHT:
        return None

    # 2) Determine the candidate class, with geometric reclassification.
    if raw_class == "tire_flat":
        if box_center_y < img_h * TIRE_MIN_Y:
            return None                                  # too high to be a tire
        candidate = "glass_shatter" if area_ratio > MAX_TIRE_AREA else "tire_flat"

    elif raw_class == "glass_shatter":
        if area_ratio > GLASS_MISSING_SINGLE and box_center_y < img_h * 0.5:
            candidate = "glass_missing"                  # single large hole
        elif confidence < MIN_GLASS_CONF or box_center_y > img_h * GLASS_MAX_Y:
            candidate = "scratch"                        # weak / low-placed glass
        else:
            candidate = "glass_shatter"

    else:
        # dent, scratch, lamp_broken, deformation, other_damage (and any
        # unforeseen class): no geometric prior, keep as-is for the conf gate.
        candidate = raw_class

    # 3) Per-class confidence gate on the final class.
    if confidence < MIN_CONF_BY_CLASS.get(candidate, DEFAULT_MIN_CONF):
        return None

    return candidate


# =============================
# DAMAGE IMAGE PIPELINE
# =============================
def analyze_damage_image(image_path, output_dir, conf=BASE_CONF):

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Failed to read image")

    img_h, img_w = img.shape[:2]

    damage_model = get_damage_model()
    infer_start = time.perf_counter()
    results = damage_model(img, conf=conf, verbose=False)
    infer_seconds = time.perf_counter() - infer_start

    valid_detections = 0
    raw_detections = 0
    glass_boxes = []
    raw_detection_counts = {}
    valid_detection_counts = {}
    valid_detection_records = []

    # =============================
    # FIRST PASS: collect boxes
    # =============================
    for r in results:
        if r.boxes is None:
            continue

        for box, cls, score in zip(
            r.boxes.xyxy,
            r.boxes.cls,
            r.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box)
            raw_class = damage_model.names[int(cls)]
            raw_detections += 1
            raw_detection_counts[raw_class] = raw_detection_counts.get(raw_class, 0) + 1

            if raw_class == "glass_shatter":
                glass_boxes.append((x1, y1, x2, y2))

    # =============================
    # AGGREGATED GLASS CHECK
    # =============================
    total_glass_area = sum(
        (x2 - x1) * (y2 - y1)
        for (x1, y1, x2, y2) in glass_boxes
    )

    glass_missing_global = (
        total_glass_area / (img_h * img_w) > GLASS_MISSING_TOTAL
    )

    # =============================
    # SECOND PASS: draw results
    # =============================
    for r in results:
        if r.boxes is None:
            continue

        for box, cls, score in zip(
            r.boxes.xyxy,
            r.boxes.cls,
            r.boxes.conf
        ):
            x1, y1, x2, y2 = map(int, box)
            raw_class = damage_model.names[int(cls)]
            confidence = float(score)

            class_name = validate_damage_class(
                raw_class,
                confidence,
                x1, y1, x2, y2,
                img_h, img_w
            )

            # Override by global reasoning
            if glass_missing_global and raw_class == "glass_shatter":
                class_name = "glass_missing"

            if class_name is None:
                continue

            valid_detections += 1
            valid_detection_counts[class_name] = valid_detection_counts.get(class_name, 0) + 1
            valid_detection_records.append({
                "bbox": [x1, y1, x2, y2],
                "raw_class": raw_class,
                "class_name": class_name,
                "confidence": confidence,
            })
            label = f"{class_name} ({confidence:.2f})"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            cv2.rectangle(
                img,
                (x1, y1 - h - 10),
                (x1 + w + 6, y1),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                img,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    output_name = image_path.stem + "_damage.jpg"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), img)

    print(f"[IMAGE] Saved to {output_path}")
    print(f"[INFO] Valid detections: {valid_detections}")
    print(f"[TIMING] damage inference: {infer_seconds * 1000:.1f} ms ({damage_model.device})")
    if glass_missing_global:
        print("🚨 GLOBAL GLASS MISSING DETECTED")

    return {
        "input_path": str(image_path),
        "output_name": output_name,
        "output_path": str(output_path),
        "image_width": img_w,
        "image_height": img_h,
        "raw_detections": raw_detections,
        "valid_detections": valid_detections,
        "raw_detection_counts": dict(sorted(raw_detection_counts.items())),
        "valid_detection_counts": dict(sorted(valid_detection_counts.items())),
        "glass_missing_global": glass_missing_global,
        "detections": valid_detection_records,
        "inference_seconds": round(infer_seconds, 4),
    }


def detect_damage_image(image_path, output_dir, conf=BASE_CONF):
    return analyze_damage_image(image_path, output_dir, conf=conf)["output_name"]
