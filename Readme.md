# Road Surveillance System

An AI-powered road surveillance platform that detects **traffic accidents** and **street fights** in real-time using a custom-trained YOLOv8 model running on Kaggle GPUs, with a Django web dashboard for monitoring and vehicle tracking.

---

## Features

### Anomaly Detection
- **Accident Detection** — Custom YOLOv8 model detects vehicle collisions with sliding-window confidence scoring
- **Fight Detection** — Identifies street fights/physical altercations from camera feeds
- **Automated Pipeline** — Detection → License plate extraction → Multi-camera vehicle tracking (all runs automatically)

### Vehicle Tracking
- **License Plate Recognition** — Forensic OCR using EasyOCR with 3× upscale + CLAHE contrast enhancement
- **Multi-Camera Search** — Parallel scanning across multiple camera feeds to build vehicle route maps
- **Fuzzy Matching** — Handles partial/blurry plate reads with configurable similarity thresholds

### Dashboard
- **Real-time Monitoring** — Live incident feed with status tracking (New → Processing → Completed)
- **Incident Details** — View snapshots, detected vehicles, license plates, and camera route maps
- **Admin Panel** — Full Django admin for managing incidents, vehicles, and search logs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER (Browser)                           │
│         Dashboard │ Tracking │ Incident Details             │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │ (AJAX polling)
             ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  DJANGO SERVER (Local)                      │
│                                                             │
│  Views:                    API Endpoints:                   │
│  • dashboard               • /api/incident/create/          │
│  • anomaly_detection        • /api/search/result/           │
│  • fight_anomaly_detection  • /api/check-incident-status/   │
│  • tracking                 • /api/check-status/            │
│  • incident_detail          • /api/incidents/               │
│                                                             │
│  Database: SQLite (Incident, Vehicle, VehicleDetection,     │
│            SearchLog, IncidentLog)                          │
└────────────┬────────────────────────────────┬───────────────┘
             │ (async request via ngrok)      │ (callback POST)
             ▼                                │
┌─────────────────────────────────────────────┴───────────────┐
│               KAGGLE SERVER (GPU)                           │
│                                                             │
│  Flask API (ngrok tunnel):                                  │
│  • /detect_incident_async  → detect + plates + track        │
│  • /search                 → multi-camera plate search      │
│  • /detect_incident        → incident detection only        │
│  • /extract_plates         → OCR plate extraction           │
│  • /track_vehicle          → single plate tracking          │
│                                                             │
│  Models: custom YOLOv8 (accident/fight) + YOLOv8m (track)   │
│  Tools:  EasyOCR, OpenCV, NumPy, PyTorch                    │
└─────────────────────────────────────────────────────────────┘
```

### Async Callback Pattern
The system uses an **async callback pattern** to bridge the local Django server with the remote Kaggle GPU server:
1. Django sends a request to Kaggle → shows a loading page immediately
2. Kaggle processes in background (detection + plate extraction + tracking)
3. Kaggle POSTs results back to Django's callback API
4. Django saves everything to the local SQLite database
5. Loading page auto-redirects when processing completes

---

## Project Structure

```
Road-Surveillance/
├── manage.py                          # Django management script
├── db.sqlite3                         # SQLite database
├── yolov8m.pt                         # YOLOv8 Medium model (tracking)
├── best (1).pt                        # Custom trained model weights
│
├── SurveillanceProject/               # Django project settings
│   ├── settings.py                    # Config (KAGGLE_API_URL, DJANGO_CALLBACK_URL)
│   ├── urls.py                        # Root URL configuration
│   ├── wsgi.py
│   └── asgi.py
│
├── SurveillanceApp/                   # Main Django application
│   ├── models.py                      # Incident, Vehicle, VehicleDetection, SearchLog, IncidentLog
│   ├── views.py                       # Page views + async Kaggle integration
│   ├── api_views.py                   # REST API endpoints (callbacks from Kaggle)
│   ├── urls.py                        # URL routing
│   ├── admin.py                       # Django admin configuration
│   └── templates/surveillance/
│       ├── dashboard.html             # Main dashboard with stats + incident list
│       ├── login.html                 # Authentication page
│       ├── details.html               # Incident detail view (snapshot + vehicle routes)
│       ├── vehicle-tracking.html      # License plate search form
│       ├── processing.html            # Loading page (vehicle tracking)
│       ├── processing_incident.html   # Loading page (anomaly detection)
│       └── tracking-result.html       # Vehicle tracking results
│
├── media/                             # Uploaded/saved media files
│   ├── incidents/                     # Incident snapshot images
│   └── detections/                    # Vehicle detection snapshots
│
├── test-data/                         # Sample test videos
│   ├── accident.mp4                   # Accident scenario
│   ├── fight1.mp4                     # Fight scenario
│   ├── normal-street.mp4              # Normal traffic (no incident)
│   ├── cam1.mp4, cam2.mp4, cam3.mp4   # Multi-camera feeds
│   └── track1.mp4, track2.mp4, ...    # Vehicle tracking test videos
│
├── new-api.ipynb                      # Kaggle notebook (latest)
├── road-surveillance-api.ipynb        # Kaggle notebook (older version)
│
└── SurveillanceEnv/                   # Python virtual environment
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip
- A Kaggle account with GPU access (for running the model)
- ngrok account (for tunneling)

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd Road-Surveillance

# Create virtual environment
python -m venv SurveillanceEnv
SurveillanceEnv\Scripts\activate        # Windows
# source SurveillanceEnv/bin/activate   # Linux/Mac

# Install Django dependencies
pip install django pillow requests
```

### 2. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. Create Superuser

```bash
python manage.py createsuperuser
```

### 4. Start Django Server

```bash
python manage.py runserver
```

### 5. Expose Django via ngrok (for Kaggle callbacks)

```bash
ngrok http 8000
```
Copy the ngrok URL and update `DJANGO_CALLBACK_URL` in `SurveillanceProject/settings.py`.

### 6. Setup Kaggle Notebook

1. Upload the notebook (`new-api.ipynb`) to Kaggle
2. Upload test videos as a Kaggle dataset
3. Upload the custom model weights (`best (1).pt`)
4. Run all cells — copy the ngrok URL printed by the tunnel
5. Update `KAGGLE_API_URL` in `SurveillanceProject/settings.py` with the Kaggle ngrok URL

### 7. Configuration

Edit `SurveillanceProject/settings.py`:

```python
KAGGLE_API_URL = "https://your-kaggle-ngrok-url.ngrok-free.dev"
DJANGO_CALLBACK_URL = "https://your-django-ngrok-url.ngrok-free.dev"
```

---

## 🔌 API Endpoints

### Django APIs (for Kaggle callbacks)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/incident/create/` | POST | Receive incident data from Kaggle |
| `/api/search/result/` | POST | Receive vehicle search results from Kaggle |
| `/api/check-incident-status/<id>/` | GET | Poll incident detection progress |
| `/api/check-status/<id>/` | GET | Poll vehicle search progress |
| `/api/incidents/` | GET | Get recent incidents list |
| `/api/incident/<id>/resolve/` | POST | Mark incident as resolved |

### Kaggle Flask APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect_incident_async` | POST | Start async anomaly detection pipeline |
| `/search` | POST | Start async multi-camera vehicle search |
| `/detect_incident` | POST | Synchronous incident detection |
| `/extract_plates` | POST | Extract license plates from video |
| `/track_vehicle` | POST | Track vehicle across cameras |
| `/status` | GET | Health check |

---

## Data Models

| Model | Description |
|-------|-------------|
| **Incident** | Detected incident (accident/fight) with snapshot, location, status |
| **Vehicle** | Vehicle linked to an incident or search, stores license plate |
| **VehicleDetection** | Where a vehicle was seen — camera name, location, timestamp, snapshot |
| **SearchLog** | User-initiated vehicle search with processing status |
| **IncidentLog** | Tracks async anomaly detection requests |

---

## Detection Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.60 | Minimum detection confidence |
| `PLATE_MATCH_CONFIDENCE` | 0.80 | Fuzzy string match threshold |
| `PRE_IMPACT_SECONDS` | 4 | Seconds before impact to scan for plates |
| `BLUR_THRESHOLD` | 80.0 | Laplacian variance threshold for blur detection |
| `IMPACT_ZONE_RADIUS` | 120 | Pixel radius around impact center |
| `DETECTION_WINDOW_SIZE` | 30 | Sliding window size for detection scoring |
| `ACTIVATION_THRESHOLD` | 15 | Minimum score within window to confirm incident |

---

## Test Data

The `test-data/` folder contains sample videos for testing:

| Video | Scenario |
|-------|----------|
| `accident.mp4` | Vehicle collision |
| `fight1.mp4` | Street fight |
| `normal-street.mp4` | Normal traffic (no incident) |
| `normal_traffic.mp4` | Normal traffic (alternate) |
| `dummy.mp4` | Dummy test video |
| `cam1.mp4` | Multi-camera surveillance feeds |
| `cam2.mp4` | Multi-camera surveillance feeds |
| `cam3.mp4` | Multi-camera surveillance feeds |
| `track1.mp4` | Vehicle tracking test feeds |
| `track2.mp4` | Vehicle tracking test feeds |
| `track3.mp4` | Vehicle tracking test feeds |

## License

This project is for educational/research purposes.
