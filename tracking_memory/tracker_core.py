from tracking_memory.car_state import CarState

cars = {}

def update_cars(detections, frame_id, fps):
    for det in detections:
        track_id = det["id"]
        center = det["center"]
        bbox = det["bbox"]

        if track_id not in cars:
            cars[track_id] = CarState(track_id)

        cars[track_id].update(center, bbox, frame_id, fps=fps)
