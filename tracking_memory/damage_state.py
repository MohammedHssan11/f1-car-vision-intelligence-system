class DamageState:
    def __init__(self):
        self.types = {}
        self.first_seen = {}
        self.last_seen = {}
        self.total_frames = {}

    def update(self, damage_type, frame_idx):
        # One crop can produce overlapping detections of the same class. They
        # are one observation at this moment, not independent confirmation of
        # persistent damage.
        if self.last_seen.get(damage_type) == frame_idx:
            return False

        self.types[damage_type] = self.types.get(damage_type, 0) + 1
        self.total_frames[damage_type] = self.total_frames.get(damage_type, 0) + 1

        if damage_type not in self.first_seen:
            self.first_seen[damage_type] = frame_idx

        self.last_seen[damage_type] = frame_idx
        return True

    def observation_count(self, damage_type):
        """Number of distinct inference frames that observed this damage."""
        return self.total_frames.get(damage_type, 0)

    def persistence_frames(self, damage_type):
        """Elapsed video-frame span from first to latest observation."""
        first = self.first_seen.get(damage_type)
        last = self.last_seen.get(damage_type)
        if first is None or last is None:
            return 0
        return last - first + 1

    def severity(self, damage_type):
        # Damage runs on a stride, so detection count is not equivalent to
        # video duration. Severity must use elapsed frames instead.
        frames = self.persistence_frames(damage_type)
        if frames > 30:
            return "HIGH"
        elif frames > 10:
            return "MEDIUM"
        else:
            return "LOW"
