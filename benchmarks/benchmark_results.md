# Phase 4 Benchmark Results

Generated: `2026-07-11T12:54:56+00:00`
Manifest: `benchmarks/manifest.json`
Machine-readable results: `benchmarks/results/benchmark_results.json`

## Evidence Boundary

Only team image labels are available in this manifest. Car boxes, damage labels, and video events are measured qualitatively only.

Expected labels and measured observations are listed separately. `null` expected values mean no ground truth label was available.

## Overall Metrics

| Metric | Value |
| ------ | ----- |
| `total_samples` | `4` |
| `completed_samples` | `4` |
| `failed_samples` | `0` |
| `image_samples` | `3` |
| `video_samples` | `1` |
| `team_labeled_image_samples` | `3` |
| `team_correct_image_samples` | `3` |
| `team_accuracy` | `1.0` |
| `car_detection_accuracy` | `None` |
| `damage_detection_accuracy` | `None` |
| `event_detection_accuracy` | `None` |
| `accuracy_notes` | `Only team image labels are available in this manifest. Car boxes, damage labels, and video events are measured qualitatively only.` |

## Measured Model Metrics (held-out)

Computed 2026-07-11 directly against labeled hold-out data, separate from this manifest's qualitative samples.

### Car detector — YOLOv8, `f1_car` (roboflow held-out val: 101 images / 111 instances)

| Metric | Value |
| ------ | ----- |
| mAP@0.5 | 0.983 |
| mAP@0.5:0.95 | 0.937 |
| Precision | 0.970 |
| Recall | 0.937 |

### Team classifier — FastAI ResNet34 (477 images, filename-derived labels — indicative)

| Team | Accuracy |
| ---- | -------- |
| Overall | 0.973 (464/477) |
| Ferrari / Renault / Williams | 1.000 |
| Red Bull Racing | 0.985 |
| McLaren | 0.984 |
| Mercedes | 0.949 |
| Racing Point | 0.944 |
| AlphaTauri | 0.828 |

Caveat: team validation labels come from filename prefixes and may overlap the classifier's training split, so treat as indicative rather than a clean held-out score. Notable confusions: AlphaTauri → Red Bull / Williams, Mercedes → Williams.

### Damage detector — YOLOv8, 7 CarDD classes

Per-class mAP not yet measured: the CarDD validation set is not currently on disk. Re-run against the CarDD YOLO `val` split to report per-class precision / recall / mAP.

## Sample Results

| Sample | Type | Expected | Observed | Metrics |
| ------ | ---- | -------- | -------- | ------- |
| `image_ferrari_0014` | `image` | `{"car_boxes":null,"damage_labels":null,"team_label":"ferrari"}` | `{"car_detection":{"class_counts":{"f1_car":1},"detection_count":1,"detections":[{"bbox":[0,0,962,639],"class_name":"f1_car","confidence":0.9454008340835571}],"max_confidence":0.9454},"damage_detection":{"glass_missing_global":false,"output_path":"benchmarks/results/images/ferrari_f1_car_0014_damage.jpg","raw_detection_counts":{"glass_shatter":1,"lamp_broken":1},"raw_detections":2,"valid_detection_counts":{"lamp_broken":1,"scratch":1},"valid_detections":2},"team_classification":{"expected_label":"ferrari","matches_expected":true,"note":"Runtime wrapper does not expose confidence; TEAM_CONF_TH may return UNKNOWN.","predicted_label":"Ferrari F1 car"}}` | `{"car_box_ground_truth_available":false,"damage_ground_truth_available":false,"team_label_available":true,"team_label_match":true}` |
| `image_mclaren_0001` | `image` | `{"car_boxes":null,"damage_labels":null,"team_label":"mclaren"}` | `{"car_detection":{"class_counts":{"f1_car":1},"detection_count":1,"detections":[{"bbox":[77,156,946,532],"class_name":"f1_car","confidence":0.9693456888198853}],"max_confidence":0.9693},"damage_detection":{"glass_missing_global":false,"output_path":"benchmarks/results/images/mclaren_f1_car_0001_damage.jpg","raw_detection_counts":{},"raw_detections":0,"valid_detection_counts":{},"valid_detections":0},"team_classification":{"expected_label":"mclaren","matches_expected":true,"note":"Runtime wrapper does not expose confidence; TEAM_CONF_TH may return UNKNOWN.","predicted_label":"McLaren F1 car"}}` | `{"car_box_ground_truth_available":false,"damage_ground_truth_available":false,"team_label_available":true,"team_label_match":true}` |
| `image_mercedes_0006` | `image` | `{"car_boxes":null,"damage_labels":null,"team_label":"mercedes"}` | `{"car_detection":{"class_counts":{"f1_car":1},"detection_count":1,"detections":[{"bbox":[75,0,824,446],"class_name":"f1_car","confidence":0.967808723449707}],"max_confidence":0.9678},"damage_detection":{"glass_missing_global":false,"output_path":"benchmarks/results/images/mercedes_f1_car_0006_damage.jpg","raw_detection_counts":{"deformation":1},"raw_detections":1,"valid_detection_counts":{"deformation":1},"valid_detections":1},"team_classification":{"expected_label":"mercedes","matches_expected":true,"note":"Runtime wrapper does not expose confidence; TEAM_CONF_TH may return UNKNOWN.","predicted_label":"Mercedes F1 car"}}` | `{"car_box_ground_truth_available":false,"damage_ground_truth_available":false,"team_label_available":true,"team_label_match":true}` |
| `video_tracker2` | `video` | `{"collisions":null,"damage_labels":null,"overtakes":null,"team_labels":null,"tracked_cars":null}` | `{"last_progress_event":{"frames_processed":597,"total_frames":597},"pipeline_summary":{"cars_tracked":25,"damage_by_car":[{"id":46,"team":"UNKNOWN","types":{"dent":1}}],"damage_type_counts":{"dent":1},"frames_processed":597,"output_size_mb":15.39,"output_video":"C:\\TERM 7\\computer vision\\final project\\outputs\\videos\\benchmark_video_tracker2.mp4","output_video_name":"benchmark_video_tracker2.mp4","processing_fps":26.7,"processing_seconds":22.39,"source_video":"C:\\TERM 7\\computer vision\\final project\\tracker\\tracker2.mp4","team_counts":{"AlphaTauri F1 car":3,"Ferrari F1 car":11,"UNKNOWN":11},"total_collisions":0,"total_overtakes":15},"progress_events_recorded":599}` | `{"damage_ground_truth_available":false,"event_ground_truth_available":false,"team_ground_truth_available":false,"track_ground_truth_available":false}` |

## Qualitative Observations

### `image_ferrari_0014`
- Car boxes are not labeled; car detection count is measured but not scored.
- Damage labels are not available; damage detections are measured but not scored.

### `image_mclaren_0001`
- Car boxes are not labeled; car detection count is measured but not scored.
- Damage labels are not available; damage detections are measured but not scored.
- No valid damage detections after post-processing.

### `image_mercedes_0006`
- Car boxes are not labeled; car detection count is measured but not scored.
- Damage labels are not available; damage detections are measured but not scored.

### `video_tracker2`
- Tracked car count has no ground truth label; value is qualitative.
- Collision/overtake counts have no event labels; values are qualitative.
- Damage counts have no frame-level labels; values are qualitative.
- Measured 25 tracks, 0 collisions, and 15 overtakes.

## Weak Classes and Retraining Needs

- video_tracker2: video pipeline produced UNKNOWN team labels; inspect crop quality or add team data.
- Add box-level car labels before reporting car detection precision/recall.
- Add damage labels before reporting damage precision/recall or per-class F1.
- Add event labels before reporting collision/overtake accuracy.
