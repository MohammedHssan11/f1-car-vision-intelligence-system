from tracking_memory.utils import iou

def assign_damage(cars, damage_detections, frame_idx):
    for damage in damage_detections:
        for car in cars.values():

            if car.last_bbox is None:
                continue

            overlap = iou(damage["bbox"], car.last_bbox)

            if overlap > 0.3:
                car.damage.update(damage["type"], frame_idx)
