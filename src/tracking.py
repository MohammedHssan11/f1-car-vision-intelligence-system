from tracking_memory.damage_state import DamageState


class CarState:
    def __init__(self, car_id):
        self.id = car_id

        self.team = None
        self.damage = DamageState()

        # =====================
        # SPEED STATE
        # =====================
        self.speed_history = []        # pixels / second
        self.accel_history = []        # pixels / second^2
        self.acceleration_samples = [] # (frame_id, pixels / second^2)
        self.speed_samples = []        # (frame_id, smoothed pixels / second)
        self.smoothed_speed = 0.0
        self._last_smoothed_speed = None
        self.velocity_px_per_frame = (0.0, 0.0)

        # =====================
        # COLLISION STATE
        # =====================
        self.collision_frames = []
        self.last_collision_frame = None

        # =====================
        # TIME & POSITION
        # =====================
        self.first_seen = None
        self.last_seen = None

        self.last_position = None
        self.last_bbox = None
        self.last_frame_id = None
        self.raw_track_ids = set()

        # =====================
        # TRAJECTORY STATE
        # =====================
        self.path_length = 0.0         # total traveled distance (pixels)

    def update(self, center, bbox, frame_id, fps=30):
        if self.last_position is not None:
            dx = center[0] - self.last_position[0]
            dy = center[1] - self.last_position[1]
            frame_gap = max(1, frame_id - self.last_frame_id)

            # =====================
            # DISTANCE (PIXELS)
            # =====================
            pixel_dist = (dx ** 2 + dy ** 2) ** 0.5

            # accumulate total path length
            self.path_length += pixel_dist

            # =====================
            # SPEED (px / sec)
            # =====================
            # A track may be briefly occluded. Dividing by the elapsed frame
            # count prevents a two-frame jump from being interpreted as one
            # frame of extreme speed.
            speed = (pixel_dist / frame_gap) * fps
            self.speed_history.append(speed)

            # =====================
            # MOTION + EXPONENTIAL SMOOTHING
            # =====================
            measured_velocity = (dx / frame_gap, dy / frame_gap)
            velocity_alpha = 0.4
            self.velocity_px_per_frame = (
                velocity_alpha * measured_velocity[0]
                + (1 - velocity_alpha) * self.velocity_px_per_frame[0],
                velocity_alpha * measured_velocity[1]
                + (1 - velocity_alpha) * self.velocity_px_per_frame[1],
            )
            alpha = 0.3
            self.smoothed_speed = (
                alpha * speed + (1 - alpha) * self.smoothed_speed
            )

            # Acceleration comes from the smoothed speed signal. Frame-level
            # detector jitter otherwise creates huge false braking spikes.
            if self._last_smoothed_speed is not None:
                elapsed_seconds = frame_gap / fps
                accel = (self.smoothed_speed - self._last_smoothed_speed) / elapsed_seconds
                self.accel_history.append(accel)
                self.acceleration_samples.append((frame_id, accel))
            self._last_smoothed_speed = self.smoothed_speed
            self.speed_samples.append((frame_id, self.smoothed_speed))

            # Kinematic history is only needed for short temporal windows.
            # Bounded storage prevents a long video from retaining every
            # frame solely for collision detection.
            history_limit = 360
            if len(self.speed_history) > history_limit:
                del self.speed_history[:-history_limit]
            if len(self.accel_history) > history_limit:
                del self.accel_history[:-history_limit]
            if len(self.acceleration_samples) > history_limit:
                del self.acceleration_samples[:-history_limit]
            if len(self.speed_samples) > history_limit:
                del self.speed_samples[:-history_limit]

        # =====================
        # UPDATE STATE
        # =====================
        self.last_position = center
        self.last_bbox = bbox
        self.last_frame_id = frame_id

        if self.first_seen is None:
            self.first_seen = frame_id

        self.last_seen = frame_id

    def set_team(self, team_name):
        if self.team is None:
            self.team = team_name
