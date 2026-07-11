# Hard Passing Test Samples

Use these exact files for a harder demo that has already passed this app.

## Recommended Photo

Path:

```text
samples/hard_test/hard_photo_kimi_raikkonen_2008_belgium_crash.jpg
uploads/images/hard_photo_kimi_raikkonen_2008_belgium_crash.jpg
```

Verified result:

```text
Car detections: 1
Team prediction: Ferrari F1 car
Damage detections: 1 lamp_broken
FastAPI upload: HTTP 200, annotated image generated
```

Source: Wikimedia Commons, "Kimi Raikkonen 2008 Belgium crash.jpg" by Mark McArdle, CC BY-SA 2.0.

## Recommended Video

Path:

```text
samples/hard_test/hard_video_2021_russian_gp_warmup.mp4
uploads/videos/hard_video_2021_russian_gp_warmup.mp4
```

Verified result:

```text
Frames processed: 538 / 538
Cars tracked: 31
Overtakes: 8
Collisions: 0
Processing speed: about 20 fps through the FastAPI background job
FastAPI job: completed
```

Source: Wikimedia Commons, "2021 Russian Grand Prix - start for warm-up.webm" by rubin16, CC BY-SA 4.0. Converted from WebM to MP4 for this app's accepted video formats.

## How To Test In The App

1. Start the app.
2. Open `http://127.0.0.1:8000/page/damage` and upload the recommended photo.
3. Open `http://127.0.0.1:8000/page/pipeline`.
4. Use this local video path:

```text
uploads/videos/hard_video_2021_russian_gp_warmup.mp4
```

5. Run it as a background job.
