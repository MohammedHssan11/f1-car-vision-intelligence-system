from fastai.vision.all import *
import pathlib
from PIL import ImageFile, UnidentifiedImageError, Image

# Allow loading truncated / incomplete images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Fix WindowsPath issue
pathlib.PosixPath = pathlib.WindowsPath

def verify_image(fname):
    try:
        img = Image.open(fname)
        img.verify()   # verify integrity
        return True
    except:
        print(f"[REMOVED BAD FILE] {fname}")
        return False

def main():
    path = Path(r"C:\TERM 7\computer vision\final project\Formula One Cars")

    # REMOVE ALL BAD IMAGES FIRST
    for f in get_image_files(path):
        if not verify_image(f):
            f.unlink()

    print("✔ All bad images removed. Starting training...")

    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(),
        num_workers=0
    )

    learn = vision_learner(dls, resnet34, metrics=[accuracy])
    learn.fine_tune(5)

    learn.export(r"C:\TERM 7\computer vision\final project\f1_team_classifier.pkl")

    print("🎉 Model trained and saved successfully!")

if __name__ == "__main__":
    main()
