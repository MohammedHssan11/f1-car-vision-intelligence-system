def detect_collisions(cars, frame_idx,
                      decel_th=-300,   # sudden deceleration
                      cooldown=15):   # frames between collisions

    collision_events = []

    for car in cars.values():

        if len(car.accel_history) < 1:
            continue

        # 1️⃣ Sudden deceleration
        sudden_brake = car.accel_history[-1] < decel_th

        # 2️⃣ New damage appears in this frame
        new_damage = any(
            first_seen == frame_idx
            for first_seen in car.damage.first_seen.values()
        )

        # cooldown to avoid duplicates
        if car.last_collision_frame is not None:
            if frame_idx - car.last_collision_frame < cooldown:
                continue

        # Collision condition
        if sudden_brake and new_damage:
            collision_events.append({
                "frame": frame_idx,
                "car_id": car.id,
                "speed": int(car.smoothed_speed),
                "damage": list(car.damage.first_seen.keys())
            })

            car.collision_frames.append(frame_idx)
            car.last_collision_frame = frame_idx

    return collision_events
