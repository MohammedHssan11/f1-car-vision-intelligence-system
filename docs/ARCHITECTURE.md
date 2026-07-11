# System Architecture Reconstruction

This document maps the current application structure to the active source files.

## High-Level System Architecture

The app is a monolithic FastAPI service that runs local computer vision inference and stores inputs/results on disk.

```mermaid
graph TD
    subgraph UI ["Browser UI"]
        A["index.html"]
        B["damage_image.html"]
        C["full_pipeline.html"]
        D["pipeline_result.html"]
    end

    subgraph API ["FastAPI App"]
        E["app/main.py"]
        F["app/api.py"]
        G["app/config.py"]
    end

    subgraph Services ["Service Wrappers"]
        H["services/damage_image.py"]
        I["services/full_pipeline.py"]
    end

    subgraph Core ["Core Vision Code"]
        J["src/detection.py"]
        K["src/pipeline.py"]
        L["src/tracking.py"]
        M["tracking_memory/*"]
    end

    subgraph Models ["Local Models"]
        N["YOLO car model: yolo_model_robflow/.../best.pt"]
        O["YOLO damage model: models/best_carDD.pt"]
        P["FastAI team model: models/f1_team_classifier.pkl"]
    end

    subgraph Storage ["Local Filesystem"]
        Q["uploads/images"]
        R["uploads/videos"]
        S["app/static/results"]
        T["outputs/videos"]
        U["tracking_memory.cars"]
    end

    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    G --> E
    G --> F
    F --> H
    F --> I
    H --> J
    I --> K
    K --> L
    K --> M
    J --> O
    K --> N
    K --> O
    K --> P
    F --> Q
    F --> R
    J --> S
    K --> T
    M --> U
```

## Request Flow

### Image Damage Flow

```mermaid
sequenceDiagram
    participant Browser
    participant API as app/api.py
    participant Config as app/config.py
    participant Service as services/damage_image.py
    participant Detector as src/detection.py
    participant FS as Local filesystem

    Browser->>API: POST /damage/image
    API->>Config: Read upload policy and output dirs
    API->>FS: Stream validated upload to uploads/images
    API->>Service: detect_damage_image(input_path, OUTPUT_IMAGE_DIR)
    Service->>Detector: detect_damage_image(...)
    Detector->>FS: Save app/static/results/*_damage.jpg
    API->>Browser: Render damage_image.html with result_image
```

### Video Pipeline Flow

```mermaid
sequenceDiagram
    participant Browser
    participant API as app/api.py
    participant Pipeline as src/pipeline.py
    participant Cars as tracking_memory.cars
    participant Models as Local models
    participant FS as Local filesystem

    Browser->>API: POST /pipeline/video or /pipeline/run
    API->>FS: Save upload or resolve safe tracker/uploads path
    API->>Pipeline: run_full_pipeline(video_path, output_name)
    Pipeline->>Cars: cars.clear()
    loop Every frame until EOF or MAX_FRAMES
        Pipeline->>Models: car_model.track(frame)
        Pipeline->>Cars: update_cars(detections, frame_idx, fps)
        Pipeline->>Models: classify team crop when team is unset
        Pipeline->>Models: run damage model every 5 frames
        Pipeline->>Cars: assign_damage, detect_collisions, detect_overtakes
        Pipeline->>FS: write annotated frame
    end
    Pipeline->>API: summary dict
    API->>Browser: Render template with summary and video path
```

## State and Tracking Memory Relationships

```mermaid
classDiagram
    class CarState {
        +int id
        +str team
        +DamageState damage
        +list speed_history
        +list accel_history
        +float smoothed_speed
        +list collision_frames
        +int last_collision_frame
        +int first_seen
        +int last_seen
        +tuple last_position
        +list last_bbox
        +float path_length
        +update(center, bbox, frame_id, fps)
        +set_team(team_name)
    }

    class DamageState {
        +dict types
        +dict first_seen
        +dict last_seen
        +dict total_frames
        +update(damage_type, frame_idx)
        +severity(damage_type)
    }

    CarState *-- DamageState
```

## External Services and Queues

* **External APIs:** None verified in the active app.
* **Queues/Workers:** None. Video processing is synchronous inside the request path.
* **Database:** None. Runtime state is in memory and persistent artifacts are files.
* **Authentication:** None.
