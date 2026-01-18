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
        self.smoothed_speed = 0.0

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

        # =====================
        # TRAJECTORY STATE
        # =====================
        self.path_length = 0.0         # total traveled distance (pixels)

    def update(self, center, bbox, frame_id, fps=30):
        if self.last_position is not None:
            dx = center[0] - self.last_position[0]
            dy = center[1] - self.last_position[1]

            # =====================
            # DISTANCE (PIXELS)
            # =====================
            pixel_dist = (dx ** 2 + dy ** 2) ** 0.5

            # accumulate total path length
            self.path_length += pixel_dist

            # =====================
            # SPEED (px / sec)
            # =====================
            speed = pixel_dist * fps
            self.speed_history.append(speed)

            # =====================
            # ACCELERATION
            # =====================
            if len(self.speed_history) > 1:
                accel = speed - self.speed_history[-2]
                self.accel_history.append(accel)

            # =====================
            # EXPONENTIAL SMOOTHING
            # =====================
            alpha = 0.3
            self.smoothed_speed = (
                alpha * speed + (1 - alpha) * self.smoothed_speed
            )

        # =====================
        # UPDATE STATE
        # =====================
        self.last_position = center
        self.last_bbox = bbox

        if self.first_seen is None:
            self.first_seen = frame_id

        self.last_seen = frame_id

    def set_team(self, team_name):
        if self.team is None:
            self.team = team_name
