from tracking_memory.car_state import CarState


def update_cars(cars, detections, frame_id, fps):
    """Update the caller-owned ``cars`` state dict from this frame's detections.

    The state dict is passed in (owned by each pipeline run) rather than being
    a module-level global, so concurrent runs — e.g. two blocking pipeline
    requests served in parallel — keep fully isolated tracking state.
    """
    for det in detections:
        track_id = det["id"]
        center = det["center"]
        bbox = det["bbox"]

        if track_id not in cars:
            cars[track_id] = CarState(track_id)

        cars[track_id].update(center, bbox, frame_id, fps=fps)
