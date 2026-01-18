def detect_overtakes(cars, frame_idx,
                     min_gap=150,
                     cooldown=40):

    events = []

    active = [
        c for c in cars.values()
        if len(c.speed_history) > 5
    ]

    # sort by progress (front first)
    active.sort(key=lambda c: c.path_length, reverse=True)

    # initialize previous rank memory
    for c in active:
        if not hasattr(c, "prev_rank"):
            c.prev_rank = None
            c.last_overtake_frame = -999

    # current ranking
    for rank, car in enumerate(active):
        car.current_rank = rank

    # detect rank swaps
    for car in active:
        if car.prev_rank is None:
            car.prev_rank = car.current_rank
            continue

        # car moved forward in ranking
        if car.current_rank < car.prev_rank:

            # cooldown
            if frame_idx - car.last_overtake_frame < cooldown:
                continue

            # overtaken car = one that lost position
            overtaken = next(
                (
                    c for c in active
                    if c.prev_rank == car.current_rank
                ),
                None
            )

            if overtaken is not None:
                events.append({
                    "frame": frame_idx,
                    "overtaker": car.id,
                    "overtaken": overtaken.id
                })

                car.last_overtake_frame = frame_idx

        car.prev_rank = car.current_rank

    return events
