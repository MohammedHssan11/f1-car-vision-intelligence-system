# Strong car-damage photo test pack

Use the files in `selected/` for hard damage-photo tests that are already proven to pass this project. These are road-car crash variants created from one public-domain source image, then filtered so every selected image has:

- at least 1 car detection
- at least 1 valid `deformation` damage detection
- a successful live `/damage/image` upload result

The broader `candidates/` and `generated_variants/` folders are evidence folders, not the pass set. Use `selected/` when you want the guaranteed-pass photos.

## Passing selected photos

| File | Hardness type | Damage result |
| --- | --- | --- |
| `selected/damage_01_original_deformation.jpg` | original public-domain crash crop | 2 `deformation` detections |
| `selected/damage_02_dark_deformation.jpg` | darker / lower light | 1 `deformation` detection |
| `selected/damage_03_rotated_deformation.jpg` | slight camera rotation | 2 `deformation` detections |
| `selected/damage_04_bright_deformation.jpg` | brighter exposure | 2 `deformation` detections |
| `selected/damage_05_jpeg_quality55_deformation.jpg` | JPEG compression | 1 `deformation` detection |
| `selected/damage_06_mild_blur_deformation.jpg` | mild blur | 1 `deformation` detection |
| `selected/damage_07_light_noise_deformation.jpg` | light sensor noise | 2 `deformation` detections |
| `selected/damage_08_gamma_low_deformation.jpg` | gamma shift, darker midtones | 1 `deformation` detection |
| `selected/damage_09_gamma_high_deformation.jpg` | gamma shift, brighter midtones | 2 `deformation` detections |
| `selected/damage_11_left_focus_deformation.jpg` | wider crop/focus variation | 1 `deformation` detection |
| `selected/damage_12_horizontal_flip_deformation.jpg` | mirrored view | 3 `deformation` detections |

Copies are also available in `uploads/images/` with the `strong_` prefix, for example `uploads/images/strong_damage_01_original_deformation.jpg`.

## Live endpoint proof

`api_upload_results.json` records the current live FastAPI run:

- tested: 11 images
- passed: 11 images
- endpoint: `POST /damage/image`
- result: each upload returned HTTP 200 and generated a JPEG in `app/static/results/`

`selected_hard_photo_results.json` records the model-level proof for the selected photos, including car count, damage count, confidence, and boxes.

## How to test manually

1. Start the app server.
2. Open `http://127.0.0.1:8000/page/damage`.
3. Upload any file from `selected/`.
4. Confirm the page shows `ANALYSIS COMPLETE` and an annotated damage image.

## Rejected / not selected

- `generated_variants/damage_10_phone_crop_deformation.jpg` was generated but rejected because it produced zero valid damage detections.
- The F1 crash candidate stayed in `candidates/`, but it was removed from `selected/` after visual QA because the damage box was not useful for road-car damage testing.
- Several downloaded Wikimedia road-car photos were kept in `candidates/` only because the current detector produced zero valid damage detections.

## Source and attribution

All selected images are derived from Wikimedia Commons `File:Car_crash_1_(cropped).jpg`, extracted from `File:Car_crash_1.jpg`; author Thue; public domain.

Source: https://commons.wikimedia.org/wiki/File:Car_crash_1_(cropped).jpg
