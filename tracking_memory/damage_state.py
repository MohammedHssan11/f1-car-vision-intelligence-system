class DamageState:
    def __init__(self):
        self.types = {}
        self.first_seen = {}
        self.last_seen = {}
        self.total_frames = {}

    def update(self, damage_type, frame_idx):
        self.types[damage_type] = self.types.get(damage_type, 0) + 1
        self.total_frames[damage_type] = self.total_frames.get(damage_type, 0) + 1

        if damage_type not in self.first_seen:
            self.first_seen[damage_type] = frame_idx

        self.last_seen[damage_type] = frame_idx

    def severity(self, damage_type):
        frames = self.total_frames.get(damage_type, 0)
        if frames > 30:
            return "HIGH"
        elif frames > 10:
            return "MEDIUM"
        else:
            return "LOW"
