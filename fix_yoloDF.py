import os
import hashlib
from pathlib import Path
import shutil

source = Path(r"C:\TERM 7\computer vision\final project\Formula One Cars")
out = Path(r"C:\TERM 7\computer vision\final project\yolo df")
out.mkdir(parents=True, exist_ok=True)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

hashes = set()

for team_folder in source.iterdir():
    if team_folder.is_dir():
        team = team_folder.name.replace(" ", "_").lower()
        i = 0
        for img in team_folder.iterdir():
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            h = file_hash(img)
            if h in hashes:
                print("Removed duplicate:", img)
                continue

            hashes.add(h)
            new_name = f"{team}_{i:04d}{img.suffix.lower()}"
            shutil.copy(img, out/new_name)
            i += 1

print("✔ Done preprocessing images!")
print("Total unique images:", len(list(out.glob('*.*'))))
 