# Road Surveillance System — Complete Project Context

> **Purpose of this document:** Give this to any AI (Gemini, ChatGPT, etc.) so it fully understands the project architecture, UI, data flow, and all template files. Then you can ask it to redesign the UI, add features, etc.

---

## 1. What This Project Does

An **AI-powered road surveillance dashboard** built with Django. It:
1. **Detects accidents and fights** from surveillance camera videos using a custom YOLOv8 model running on Kaggle (GPU)
2. **Extracts license plates** from vehicles involved using EasyOCR
3. **Tracks vehicles** across multiple camera feeds using fuzzy string matching
4. **Displays results** on a real-time web dashboard with snapshots, timelines, and route maps

---

## 2. Architecture (Two Servers)

```
YOUR PC (Django)                           KAGGLE (Flask + GPU)
┌────────────────────┐   ngrok tunnels    ┌─────────────────────┐
│ Django Server:8000 │◄──────────────────►│ Flask Server:5001   │
│ - Dashboard UI     │  POST requests     │ - YOLOv8 models     │
│ - SQLite Database  │  + callbacks       │ - EasyOCR            │
│ - User Auth        │                    │ - Vehicle tracking   │
└────────────────────┘                    └─────────────────────┘
```

- **Django (local):** Serves the web UI, handles authentication, stores data in SQLite
- **Flask (Kaggle):** Runs AI models on GPU, processes videos, sends results back via HTTP POST callbacks
- **ngrok:** Creates public URLs for both servers so they can communicate over the internet

---

## 3. Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Django 5.2 |
| Database | SQLite3 |
| Frontend | HTML, CSS (vanilla), JavaScript (vanilla) |
| Icons | Bootstrap Icons CDN |
| AI Backend | Flask, PyTorch, Ultralytics YOLOv8, EasyOCR, OpenCV |
| Tunneling | ngrok (pyngrok) |

**No CSS frameworks** (no Bootstrap, no Tailwind). All styles are inline `<style>` blocks within each template. No shared base template is used — each page is standalone.

---

## 4. Database Models (models.py)

```python
class Incident(models.Model):
    incident_type = models.CharField(choices=[('accident','Accident'),('fight','Fight')])
    location = models.CharField(max_length=255, default="Unknown")
    timestamp = models.CharField(max_length=100, default="Unknown")
    status = models.CharField(choices=[('new','New'),('processing','Processing'),
                                       ('completed','Completed'),('reviewed','Reviewed')])
    snapshot = models.ImageField(upload_to='incidents/', null=True)
    resolved = models.BooleanField(default=False)
    detected_at = models.DateTimeField(auto_now_add=True)

class Vehicle(models.Model):
    license_plate = models.CharField(max_length=20)
    detection_confidence = models.FloatField(default=0.0)
    total_ocr_scans = models.IntegerField(default=0)
    incident = models.ForeignKey(Incident, related_name='vehicles', null=True)
    search_log = models.ForeignKey('SearchLog', related_name='vehicles', null=True)

class VehicleDetection(models.Model):
    vehicle = models.ForeignKey(Vehicle, related_name='detections')
    camera_name = models.CharField(max_length=100)
    camera_location = models.CharField(max_length=255)
    timestamp = models.CharField(max_length=100)
    matched_text = models.CharField(max_length=50)
    snapshot = models.ImageField(upload_to='detections/', null=True)
    detected_at = models.DateTimeField(auto_now_add=True)

class SearchLog(models.Model):
    license_plate = models.CharField(max_length=20)
    is_processed = models.BooleanField(default=False)
    searched_at = models.DateTimeField(auto_now_add=True)

class IncidentLog(models.Model):
    video_path = models.CharField(max_length=500)
    is_processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
```

**Relationships:**
- Incident → has many Vehicles
- Vehicle → has many VehicleDetections (sightings across cameras)
- SearchLog → has many Vehicles

---

## 5. URL Routing

### Pages (all require @login_required)
| URL | View Function | Template | Purpose |
|-----|--------------|----------|---------|
| `/` | `dashboard` | `dashboard.html` | Main dashboard — stats cards, incident list, video selector dropdown, detection buttons |
| `/login/` | `login_view` | `login.html` | Login form with username/password |
| `/anomaly-detection/` | `anomaly_detection` | (redirects) | POST-only: triggers accident detection on selected video |
| `/fight-anomaly-detection/` | `fight_anomaly_detection` | (redirects) | POST-only: triggers fight detection on selected video |
| `/anomaly-detection/processing/<id>/` | `incident_processing` | `processing_incident.html` | Loading screen while AI processes video |
| `/incident/<id>/` | `incident_detail` | `details.html` | Incident detail — snapshot, vehicles, route timeline |
| `/tracking/` | `tracking` | `vehicle-tracking.html` | License plate search form + recent searches |
| `/tracking/processing/<id>/` | `processing` | `processing.html` | Loading screen while tracking vehicle |
| `/tracking/<id>/` | `tracking_results` | `tracking-result.html` | Tracking results — route map with camera sightings |

### API Endpoints (JSON)
| URL | Method | Purpose |
|-----|--------|---------|
| `/api/incidents/` | GET | Returns recent incidents list (for auto-refresh) |
| `/api/incident/create/` | POST | **Callback from Kaggle** — creates Incident + Vehicles in DB |
| `/api/incident/<id>/resolve/` | POST | Marks incident as resolved |
| `/api/search/result/` | POST | **Callback from Kaggle** — saves tracking results |
| `/api/check-status/<id>/` | GET | **Polling** — returns {is_processed: true/false} for tracking |
| `/api/check-incident-status/<id>/` | GET | **Polling** — returns {is_processed: true/false} for detection |

---

## 6. Settings (settings.py) — Key Config

```python
KAGGLE_API_URL = "https://xxx.ngrok-free.dev"       # Kaggle Flask server URL
DJANGO_CALLBACK_URL = "https://yyy.ngrok-free.dev"  # Django callback URL

KAGGLE_VIDEO_BASE_PATH = "/kaggle/input/datasets/akhilsiji21/roadsurveillance-testdata"
AVAILABLE_VIDEOS = [
    {"name": "Accident Video", "file": "accident.mp4", "type": "accident"},
    {"name": "Fight Video", "file": "fight1.mp4", "type": "fight"},
    {"name": "Normal Traffic", "file": "normal_traffic.mp4", "type": "accident"},
    # ... more videos
]
```

---

## 7. Complete Request Flow (Async Callback Pattern)

### Flow 1: Accident/Fight Detection
```
User clicks "Run Accident Detection" on dashboard
    ↓
Django creates IncidentLog (is_processed=False)
Django spawns background thread → POST to Kaggle /detect_incident_async
Django redirects to processing_incident.html (loading screen)
    ↓
Browser polls /api/check-incident-status/<id>/ every 5 seconds
    ↓
Kaggle processes video (1-5 minutes):
  Phase 1: YOLO scans frames → detects accident/fight
  Phase 2: OCR extracts license plates from bumper crops
  Phase 3: Searches all cameras for detected plates
    ↓
Kaggle POSTs results to Django /api/incident/create/
Django saves Incident + Vehicles + VehicleDetections
Django sets IncidentLog.is_processed = True
    ↓
Next browser poll sees is_processed=True → redirects to dashboard
Dashboard shows the new incident with "🔴 JUST NOW" badge (green glow)
```

### Flow 2: Vehicle Tracking
```
User types plate "JV316S" on vehicle-tracking.html → clicks "Track Vehicle"
    ↓
Django creates SearchLog (is_processed=False)
Django spawns background thread → POST to Kaggle /search
Django redirects to processing.html (loading screen with plate displayed)
    ↓
Browser polls /api/check-status/<id>/ every 3 seconds
    ↓
Kaggle spawns one thread per camera → scans all feeds for matching plate
    ↓
Kaggle POSTs results to Django /api/search/result/
Django saves Vehicle + VehicleDetections
Django sets SearchLog.is_processed = True
    ↓
Next browser poll sees is_processed=True → redirects to tracking-result.html
Shows route map timeline with camera sightings
```

---

## 8. All Template Files — Current UI Description

### 8.1 login.html
- **Layout:** Centered card on blue gradient background
- **Elements:** Logo image, "Surveillance System" title, username/password fields, login button
- **Footer:** 3 feature icons (Real-time Monitoring, Vehicle Tracking, Instant Alerts)
- **Colors:** Blue gradient (#2291c1 → #71baeb), white card, black text
- **Static file:** Uses `{% static 'img/logo.png' %}` for logo

### 8.2 dashboard.html (MAIN PAGE)
- **Navbar:** Blue gradient, "Surveillance System" brand, Dashboard/Vehicle Tracking/Logout links
- **Stats Section:** 4 stat cards in a row (Total Incidents, Accidents, Fights, New Alerts) with gradient backgrounds
- **Video Selector:** Dropdown `<select>` listing all AVAILABLE_VIDEOS from settings
- **Action Buttons:** "Run Fight Detection" and "Run Accident Detection" (red gradient), "Refresh"
- **Incident List:** Cards with left-border color coding:
  - Status "new" → green left border
  - First "new" incident → **"latest" class** with green glow animation + "🔴 JUST NOW" badge
  - Each card shows: incident type badge, location, time, vehicle count, status badge, "View Details" button
- **JavaScript:** Auto-refresh every 30 seconds, video dropdown syncs to both form hidden inputs
- **Empty State:** Search icon + "No Incidents Detected" message

### 8.3 details.html (INCIDENT DETAIL)
- **Back button:** "← Back to Dashboard" link
- **Incident Header:** Type badge (accident=purple gradient, fight=yellow-green gradient), incident ID, metadata grid (location, timestamp, detected at, status), "Mark as Resolved" button
- **Content Grid:** 2 columns
  - Left: Incident snapshot image
  - Right: Vehicles involved list (plate number in gradient text, confidence, OCR scans, camera count)
- **Route Map:** For each vehicle, shows a vertical timeline:
  - Numbered markers (1, 2, 3...)
  - Camera name, location, timestamp, matched text
  - Detection snapshot image
- **JavaScript:** resolveIncident() sends POST to /api/incident/<id>/resolve/

### 8.4 vehicle-tracking.html
- **Search Section:** Centered search icon, "Vehicle Tracking System" title, license plate input (monospace, centered, uppercase, letter-spacing), "Track Vehicle" button
- **Info Box:** Blue callout explaining what the system will do
- **Recent Searches:** List of previous searches with plate number + date + "View Results →" link
- **JavaScript:** Auto-uppercase input, loading spinner on submit

### 8.5 processing.html (VEHICLE TRACKING LOADING)
- **Layout:** Centered white card, full-viewport centered
- **Elements:** Blue spinning circle, large plate display (gradient text, monospace), "Analyzing Footage" title
- **Camera List:** Hardcoded 3 cameras (Main Street, Highway Exit 42, Downtown Plaza) with pulsing "Scanning" badges
- **Progress Bar:** Animated gradient fill (blue → light blue)
- **JavaScript:** Polls /api/check-status/<id>/ every 3 seconds → redirects to /tracking/<id>/ when done

### 8.6 processing_incident.html (DETECTION LOADING)
- **Layout:** Same centered card pattern
- **Elements:** Red spinning circle, "Analyzing Video Feed" title
- **Steps List:** 3 steps with pulsing badges:
  1. Incident Detection (YOLOv8) — "Processing"
  2. License Plate Extraction — "Pending"
  3. Multi-Camera Vehicle Tracking — "Pending"
- **Progress Bar:** Animated gradient fill (red → orange)
- **JavaScript:** Polls /api/check-incident-status/<id>/ every 5 seconds → redirects to / when done

### 8.7 tracking-result.html
- **Header:** Large plate display (gradient text), search date/time
- **Summary Cards:** Incidents Found count, Total Detections count
- **Route Timeline:** Vertical timeline with numbered circle markers:
  - Camera name, location, time, matched text, date
  - Detection snapshot image
- **Incident Link:** Yellow callout if vehicle was involved in an incident, with link to incident detail
- **Not Found State:** Large search icon, "Vehicle Not Found" with possible reasons list, "Try Another Search" button

---

## 9. Current Design System

### Color Palette
| Usage | Color | Hex |
|-------|-------|-----|
| Primary (navbar, buttons, links) | Blue gradient | #2291c1 → #71baeb |
| Danger (detection buttons, incident badge) | Red gradient | #ff6b6b → #ee5a52 |
| Success (new status, latest glow) | Green | #43e97b |
| Accident badge | Purple gradient | #bb73c3 → #cb4b5c |
| Fight badge | Yellow-green gradient | #d1d141 → #a4c500 |
| Background | Light gray | #f5f7fa |
| Cards | White | #ffffff |
| Text primary | Dark gray | #333333 |
| Text secondary | Medium gray | #666666 |
| Borders | Light gray | #e0e0e0 |

### Typography
- **Font:** Segoe UI, Tahoma, Geneva, Verdana, sans-serif (system fonts)
- **Plate numbers:** Monospace, bold, gradient text
- **No Google Fonts imported**

### Component Patterns
- **Cards:** White background, border-radius: 7px, box-shadow: 0 2px 10px rgba(0,0,0,0.08)
- **Buttons:** Gradient backgrounds, border-radius: 7px, hover: translateY(-2px)
- **Badges:** Small pills with border-radius: 13px and bright colors
- **Inputs:** 2px solid #e0e0e0 border, focus: blue border + box-shadow
- **Icons:** Bootstrap Icons via CDN

### Layout
- **No CSS framework** — all vanilla CSS with inline `<style>` blocks
- **No base template** — each page is completely standalone HTML
- **Responsive:** Some media queries for mobile, but limited
- **Max-widths:** Dashboard 1200px, tracking 1000px, details 1200px

---

## 10. File Structure

```
Road-Surveillance/
├── SurveillanceProject/
│   ├── settings.py          # Django config, KAGGLE_API_URL, AVAILABLE_VIDEOS
│   ├── urls.py              # Root URL config (includes SurveillanceApp.urls)
│   └── wsgi.py
├── SurveillanceApp/
│   ├── models.py            # Incident, Vehicle, VehicleDetection, SearchLog, IncidentLog
│   ├── views.py             # Page views (dashboard, detection triggers, tracking)
│   ├── api_views.py         # JSON API endpoints (callbacks, polling)
│   ├── urls.py              # URL routing for all pages and APIs
│   ├── admin.py             # Django admin config
│   ├── static/
│   │   └── img/logo.png     # Logo used on login page
│   └── templates/surveillance/
│       ├── login.html
│       ├── dashboard.html
│       ├── details.html
│       ├── vehicle-tracking.html
│       ├── processing.html
│       ├── processing_incident.html
│       └── tracking-result.html
├── media/                   # Uploaded snapshots (incidents/, detections/)
├── db.sqlite3               # SQLite database
├── new-api.ipynb            # Kaggle notebook (Flask server + AI models)
└── manage.py
```

---

## 11. Known Limitations / Areas for Improvement

1. **No shared base template** — Every page has its own full HTML/CSS, leading to duplicated navbar code and inconsistent styles across pages (e.g., navbar text is black on dashboard but white on details page)
2. **No CSS framework** — All styling is custom vanilla CSS inline in each template
3. **Hardcoded camera list on processing.html** — The tracking loading page shows 3 hardcoded camera names instead of dynamically pulling from the server
4. **No dark mode** support
5. **Limited responsive design** — Works on desktop but could improve mobile layout
6. **No notification system** — No sound or visual notification when a new incident is detected
7. **No filtering/search** on the incident list — Can't filter by type, date, status
8. **Stats cards don't link** to filtered views
9. **No pagination** on incident list (currently shows last 20)
10. **Resolve button** uses `alert()` instead of a toast/notification

---

## 12. How to Use This Document

Paste this entire document into Gemini (or any AI) and then ask things like:
- "Redesign the dashboard to be more modern with dark mode"
- "Add a base template to share the navbar across all pages"
- "Make the processing pages show real-time progress from the Kaggle server"
- "Add filters and pagination to the incident list"
- "Make the UI fully responsive for mobile"
- "Add charts/graphs showing incident trends over time"
