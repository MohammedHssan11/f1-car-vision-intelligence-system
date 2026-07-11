# tracking_memory/utils.py

def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0
