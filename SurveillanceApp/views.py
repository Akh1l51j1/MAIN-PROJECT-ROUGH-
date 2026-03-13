# SurveillanceApp\views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.conf import settings
from .models import Incident, Vehicle, SearchLog, IncidentLog
import requests
import threading
import base64
from io import BytesIO
from django.core.files.base import ContentFile
from PIL import Image
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count
from django.db.models.functions import TruncDate
import json


def login_view(request):
    """Render and process the login form.

    GET  → display the login page.
    POST → authenticate credentials; on success redirect to dashboard,
           on failure add an error message and re-render the form.
    Already-authenticated users are redirected straight to the dashboard
    so they never see the login screen.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        user = authenticate(
            request,
            username=request.POST.get('username'),
            password=request.POST.get('password')
        )
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password')

    return render(request, 'surveillance/login.html')


@login_required(login_url='login')
def dashboard(request):
    """Render the main dashboard with live incident statistics.

    Passes two context items to the template:
      - stats: aggregate counts (total, by type, and unreviewed 'new' alerts).
      - incidents: the 10 most recently detected incidents for the activity feed.
    """
    stats = {
        'total': Incident.objects.count(),
        'accidents': Incident.objects.filter(incident_type='accident').count(),
        'fights': Incident.objects.filter(incident_type='fight').count(),
        'new_alerts': Incident.objects.filter(status='new').count()
    }

    incidents = Incident.objects.all().order_by('-detected_at')[:20]

    # Calculate trends for the last 7 days
    sevendays_ago = timezone.now() - timedelta(days=6) # 6 days ago + today = 7 days
    
    # Query database for daily counts
    daily_counts = (Incident.objects
                    .filter(detected_at__gte=sevendays_ago)
                    .annotate(day=TruncDate('detected_at'))
                    .values('day')
                    .annotate(count=Count('id'))
                    .order_by('day'))
                    
    # Initialize dictionary to ensure all 7 days have a value (even 0)
    trend_dict = {(timezone.now() - timedelta(days=i)).date().strftime('%b %d'): 0 for i in range(6, -1, -1)}
    
    for entry in daily_counts:
        day_str = entry['day'].strftime('%b %d')
        if day_str in trend_dict:
            trend_dict[day_str] = entry['count']
            
    chart_labels = json.dumps(list(trend_dict.keys()))
    chart_data = json.dumps(list(trend_dict.values()))

    return render(request, 'surveillance/dashboard.html', {
        'incidents': incidents,
        'stats': stats,
        'fight_videos': [v for v in settings.AVAILABLE_VIDEOS if v['type'] == 'fight'],
        'accident_videos': [v for v in settings.AVAILABLE_VIDEOS if v['type'] == 'accident'],
        'chart_labels': chart_labels,
        'chart_data': chart_data,
    })


def send_detection_to_kaggle_async(log_id, video_path):
    """Send a video analysis request to the Kaggle Flask API (runs in a background thread).

    Posts to /detect_incident_async with:
      - incident_log_id: the DB primary key of the IncidentLog record,
        so the Kaggle service can reference it in its callback.
      - video_path: path to the video file on the Kaggle filesystem.
      - callback_url: the Django endpoint that receives the finished result.

    On network error or a non-200 response the log is marked as processed
    immediately so the frontend polling loop does not hang indefinitely.
    """
    try:
        response = requests.post(
            f"{settings.KAGGLE_API_URL}/detect_incident_async",
            json={
                "incident_log_id": log_id,
                "video_path": video_path,
                "callback_url": f"{settings.DJANGO_CALLBACK_URL}/api/incident/create/"
            },
            timeout=30 
        )
        
        if response.status_code != 200:
            print(f"Kaggle API Error: {response.status_code} - {response.text}")
            # Mark as processed so the waiting-room page does not spin forever.
            IncidentLog.objects.filter(id=log_id).update(is_processed=True)
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Kaggle API: {str(e)}")
        IncidentLog.objects.filter(id=log_id).update(is_processed=True)


@login_required(login_url='login')
def anomaly_detection(request):
    """Trigger accident detection on the designated road-surveillance video.

    On POST:
      1. Create an IncidentLog record (is_processed=False) to track this job.
      2. Spawn a daemon thread that calls send_detection_to_kaggle_async,
         keeping the POST-response latency low.
      3. Redirect to the 'incident_processing' waiting-room view.

    The daemon thread is used so the HTTP request completes immediately;
    the thread is killed automatically when the server process exits.
    """
    if request.method == 'POST':
        selected_file = request.POST.get('video_file', 'accident.mp4')
        # Validate: only allow files from the configured list
        allowed_files = [v['file'] for v in settings.AVAILABLE_VIDEOS]
        if selected_file not in allowed_files:
            selected_file = 'accident.mp4'
        video_path = f"{settings.KAGGLE_VIDEO_BASE_PATH}/{selected_file}"
        
        # Create a log entry to track this detection job's processing state.
        log = IncidentLog.objects.create(
            video_path=video_path,
            is_processed=False
        )
        
        thread = threading.Thread(
            target=send_detection_to_kaggle_async,
            args=(log.id, video_path)
        )
        thread.daemon = True
        thread.start()
        
        return redirect('incident_processing', log_id=log.id)
    
    return redirect('dashboard')


@login_required(login_url='login')
def fight_anomaly_detection(request):
    """Trigger fight/violence detection on the designated surveillance video.

    Identical flow to anomaly_detection but points at the fight test video.
    A separate view is used so the two detection types can evolve
    independently (e.g. different confidence thresholds, video sources).
    """
    if request.method == 'POST':
        selected_file = request.POST.get('video_file', 'fight1.mp4')
        allowed_files = [v['file'] for v in settings.AVAILABLE_VIDEOS]
        if selected_file not in allowed_files:
            selected_file = 'fight1.mp4'
        video_path = f"{settings.KAGGLE_VIDEO_BASE_PATH}/{selected_file}"
        
        log = IncidentLog.objects.create(
            video_path=video_path,
            is_processed=False
        )
        
        thread = threading.Thread(
            target=send_detection_to_kaggle_async,
            args=(log.id, video_path)
        )
        thread.daemon = True
        thread.start()
        
        return redirect('incident_processing', log_id=log.id)
    
    return redirect('dashboard')


@login_required(login_url='login')
def incident_processing(request, log_id):
    """Waiting-room view shown while the Kaggle API processes an incident.

    Renders a loading screen that the frontend JavaScript polls via
    /api/incident/status/<log_id>/ (check_incident_status_api).
    Once the Kaggle callback fires and sets is_processed=True, the next
    poll redirects the user automatically to the dashboard.
    """
    log = get_object_or_404(IncidentLog, id=log_id)
    
    # If results have already arrived (e.g. user refreshed after completion),
    # skip the waiting room and go straight to the dashboard.
    if log.is_processed:
        return redirect('dashboard')
    
    return render(request, 'surveillance/processing_incident.html', {
        'log': log
    })


@login_required(login_url='login')
def incident_detail(request, incident_id):
    """Render the detail page for a single confirmed incident.

    Builds a route list for each involved vehicle by iterating its
    VehicleDetection records in chronological order. Each detection
    entry contains camera metadata, timestamp, OCR plate text, and
    a snapshot image reference — displayed as a timeline in the template.
    """
    incident = get_object_or_404(Incident, id=incident_id)
    
    if incident.status == 'new':
        incident.status = 'reviewed'
        incident.save(update_fields=['status'])
        
    vehicles = incident.vehicles.all()

    for v in vehicles:
        v.route = [
            {
                'camera': {
                    'name': d.camera_name,
                    'location': d.camera_location
                },
                'timestamp': d.timestamp,
                'matched_text': d.matched_text,
                'snapshot': d.snapshot,
                'detected_at': d.detected_at
            }
            for d in v.detections.all().order_by('detected_at')
        ]

    return render(request, 'surveillance/details.html', {
        'incident': incident,
        'vehicles': vehicles
    })


def send_to_kaggle_async(search_id, license_plate):
    """Send a plate-search request to the Kaggle Flask API (runs in a background thread).

    Posts to /search with:
      - search_id: the DB primary key of the SearchLog record.
      - license_plate: the uppercased plate string to look for.
      - callback_url: the Django endpoint that receives the scan results.

    A generous 180-second timeout is used because the scan must iterate
    through potentially hours of footage across multiple camera feeds.
    On failure the search log is immediately marked as processed to
    unblock the frontend polling loop.
    """
    try:
        response = requests.post(
            f"{settings.KAGGLE_API_URL}/search",
            json={
                "search_id": search_id,
                "license_plate": license_plate,
                "callback_url": f"{settings.DJANGO_CALLBACK_URL}/api/search/result/"
            },
            timeout=180 
        )
        
        if response.status_code != 200:
            print(f" Kaggle API Error: {response.status_code} - {response.text}")
            SearchLog.objects.filter(id=search_id).update(is_processed=True)
            
    except requests.exceptions.RequestException as e:
        print(f" Failed to connect to Kaggle API: {str(e)}")
        SearchLog.objects.filter(id=search_id).update(is_processed=True)


@login_required(login_url='login')
def tracking(request):
    """Accept a license plate from the search form and launch a cross-camera scan.

    GET  → render the vehicle-tracking search form with the 5 most recent searches.
    POST → normalise the plate (uppercase, strip whitespace), create a SearchLog,
           fire a daemon thread to call send_to_kaggle_async, then redirect to
           the 'processing' waiting-room view.
    """
    """
    Starts a background task to trace a target vehicle based on the provided license plate.
    """
    if request.method == 'POST':
        plate = request.POST.get('license_plate').upper().strip()
        
        # Create a log entry to track this search job's processing state.
        search = SearchLog.objects.create(
            license_plate=plate,
            is_processed=False
        )
        
        thread = threading.Thread(
            target=send_to_kaggle_async,
            args=(search.id, plate)
        )
        thread.daemon = True
        thread.start()
        
        return redirect('processing', search_id=search.id)

    return render(request, 'surveillance/vehicle-tracking.html', {
        'recent_searches': SearchLog.objects.order_by('-searched_at')[:5]
    })


@login_required(login_url='login')
def processing(request, search_id):
    """Waiting-room view shown while the Kaggle API scans cameras for a plate.

    Renders a loading screen polled by /api/search/status/<search_id>/.
    Once the Kaggle callback sets is_processed=True, the user is
    redirected to the tracking_results view with the full sighting data.
    """
    search = get_object_or_404(SearchLog, id=search_id)
    
    # If already complete (e.g. user refreshed), skip the waiting room.
    if search.is_processed:
        return redirect('tracking_results', search_id=search_id)
    
    return render(request, 'surveillance/processing.html', {
        'search': search
    })


@login_required(login_url='login')
def tracking_results(request, search_id):
    """Render the results page showing where and when a vehicle was spotted.

    Builds route_data — a list of per-vehicle dictionaries containing the
    vehicle record, its linked incident (if any), and an ordered list of
    camera detections. The template uses this to render a timeline and map.
    """
    search = get_object_or_404(SearchLog, id=search_id)
    vehicles = search.vehicles.all()

    # Construct a structured route for each matching vehicle, ordered
    # chronologically so the template can render a clear movement timeline.
    route_data = [
        {
            'vehicle': v,
            'incident': v.incident,
            'detections': [
                {
                    'camera': {
                        'name': d.camera_name,
                        'location': d.camera_location
                    },
                    'timestamp': d.timestamp,
                    'matched_text': d.matched_text,
                    'snapshot': d.snapshot,
                    'detected_at': d.detected_at
                }
                for d in v.detections.all().order_by('detected_at')
            ]
        }
        for v in vehicles
    ]

    return render(request, 'surveillance/tracking-result.html', {
        'search': search,
        'found': vehicles.exists(), # Boolean flag to show/hide "no results" message.
        'route_data': route_data
    })