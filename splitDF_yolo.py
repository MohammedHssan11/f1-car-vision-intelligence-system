import os
import random
import shutil

# Paths
source = (r"C:\TERM 7\computer vision\final project\yolo df")

output = "splited_dataset"

# Train/Val split
train_split = 0.8

# Create folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(output, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output, split, "labels"), exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

# Split
train_count = int(len(images) * train_split)
train_images = images[:train_count]
val_images = images[train_count:]

# Move files
for img in train_images:
    shutil.copy(os.path.join(source, img), os.path.join(output, "train", "images", img))

for img in val_images:
    shutil.copy(os.path.join(source, img), os.path.join(output, "val", "images", img))

print("Done! 🎯")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
