"""Vehicle class mapping: COCO -> Indonesian categories.

YOLOv8 pretrained on COCO outputs 80 classes. We map the relevant
vehicle classes to the local categories used by DISHUB enforcement.
Angkot detection from generic 'car/bus' is approximate; Phase 3
fine-tuning on Jakarta-specific footage is required for accuracy.
"""

# COCO class id -> (local label, English label)
VEHICLE_CLASSES = {
    1: ("sepeda", "bicycle"),
    2: ("mobil", "car"),
    3: ("motor", "motorcycle"),
    5: ("bus", "bus"),
    7: ("truk", "truck"),
}

VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())


def coco_to_local(coco_id: int) -> str | None:
    entry = VEHICLE_CLASSES.get(coco_id)
    return entry[0] if entry else None


def is_vehicle(coco_id: int) -> bool:
    return coco_id in VEHICLE_CLASS_IDS
