# SurveillanceApp\api_views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from .models import Incident, Vehicle, VehicleDetection, SearchLog, IncidentLog
import base64
import json


@csrf_exempt
def create_incident_api(request):
    """Receive and persist a completed incident detection result from the Kaggle API.
    Called by the Kaggle Flask service after the 3-phase pipeline finishes.
   
    Processing steps:
      1. If status == 'no_incident', skip DB writes and just mark the log done.
      2. Create an Incident record and optionally attach the base64 snapshot.
      3. For each vehicle in the payload, create a Vehicle and its
         VehicleDetection records, saving per-camera snapshots to disk.
      4. Mark the IncidentLog as processed so the frontend polling loop exits.

    The except block always marks the log as processed so the user is never
    left stuck on the loading screen due to a data error.
    """
    try:
        data = json.loads(request.body)
        incident_log_id = data.get('incident_log_id')

        # Early exit: Kaggle found nothing — just unlock the UI.
        if data.get('status') == 'no_incident':
            if incident_log_id:
                IncidentLog.objects.filter(id=incident_log_id).update(is_processed=True)
            return JsonResponse({'success': True, 'message': 'No incident detected'})

        # Create the top-level Incident record.
        incident = Incident.objects.create(
            incident_type=data['incident_type'],
            location=data.get('location', 'Main Feed'),
            status='new'
        )

        # Decode and save the incident overview snapshot if one was provided.
        if data.get('snapshot'):
            image_data = data['snapshot']
            # Strip the data URI prefix (e.g. "data:image/jpeg;base64,") if present.
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            incident.snapshot.save(
                'incident.jpg',
                ContentFile(base64.b64decode(image_data))
            )

        # Persist each involved vehicle and its per-camera detection records.
        for v in data.get('vehicles', []):
            vehicle = Vehicle.objects.create(
                incident=incident,
                license_plate=v['plate'],
                detection_confidence=v.get('confidence', 0.0)
            )

            # Each detection represents one camera sighting of this vehicle.
            for d in v.get('detections', []):
                vd = VehicleDetection.objects.create(
                    vehicle=vehicle,
                    camera_name=d['camera'],
                    camera_location=d['location'],
                    timestamp=d['timestamp'],
                    matched_text=d['matched_text']
                )

                # Decode and save the per-camera snapshot if one was provided.
                if d.get('snapshot'):
                    image_data = d['snapshot']
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    vd.snapshot.save(
                        'camera.jpg',
                        ContentFile(base64.b64decode(image_data))
                    )

        # Signal the frontend that processing is complete.
        if incident_log_id:
            IncidentLog.objects.filter(id=incident_log_id).update(is_processed=True)

        return JsonResponse({'success': True, 'incident_id': incident.id})
    
    except Exception as e:
        print(f"Error in create_incident_api: {str(e)}")
        # Always mark as processed on failure so the frontend does not spin forever.
        if 'incident_log_id' in locals() and incident_log_id:
            IncidentLog.objects.filter(id=incident_log_id).update(is_processed=True)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
def search_result_api(request):
    """Receive and persist the results of a cross-camera plate search from Kaggle.

    Called by the Kaggle Flask service after scan_camera_feed threads finish.
    Processing steps:
      1. If detections is empty, skip Vehicle creation (vehicle was not found).
      2. Create a Vehicle linked to the SearchLog and save each camera
         detection with its snapshot image.
      3. Mark the SearchLog as processed so the polling loop redirects to results.

    The except block always marks the search as processed to unblock the UI.
    """
    try:
        data = json.loads(request.body)
        
        search_id = data.get('search_id')
        detections = data.get('detections', [])
        
        # Only create a Vehicle record if at least one camera sighting was found.
        if detections:
            vehicle = Vehicle.objects.create(
                search_log_id=search_id,
                license_plate=data['plate'],
                detection_confidence=data.get('confidence', 0.0)
            )

            for d in detections:
                vd = VehicleDetection.objects.create(
                    vehicle=vehicle,
                    camera_name=d['camera'],
                    camera_location=d['location'],
                    timestamp=d['timestamp'],
                    matched_text=d['matched_text']
                )

                # Decode and save the per-camera snapshot if one was provided.
                if d.get('snapshot'):
                    image_data = d['snapshot']
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    
                    vd.snapshot.save(
                        f"{d['camera']}_detection.jpg",
                        ContentFile(base64.b64decode(image_data))
                    )
        
        # Mark as processed regardless of whether a match was found,
        # so the 'processing' waiting-room page always redirects to results.
        SearchLog.objects.filter(id=search_id).update(is_processed=True)
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        print(f" Error in search_result_api: {str(e)}")
        # Always mark as processed on failure so the frontend is not left waiting.
        if 'search_id' in locals():
            SearchLog.objects.filter(id=search_id).update(is_processed=True)
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
def check_status_api(request, search_id):
    """Return the processing status of a plate-search job (polled by the frontend).

    The 'processing' waiting-room template calls this endpoint on an interval.
    Returns { is_processed: bool, search_id: int }.
    Returns 404 with an error message if the SearchLog does not exist.
    """
    try:
        search = SearchLog.objects.get(id=search_id)
        return JsonResponse({
            'is_processed': search.is_processed,
            'search_id': search_id
        })
    except SearchLog.DoesNotExist:
        return JsonResponse({
            'is_processed': False,
            'error': 'Search not found'
        }, status=404)


def get_incidents_api(request):
    """Return a JSON list of the 10 most recent incidents for the dashboard feed.

    Used by any frontend component that needs a lightweight incident list
    without a full page reload (e.g. a live-updating alert ticker).
    Returns id, type, and status for each incident.
    """
    return JsonResponse({
        'incidents': [
            {
                'id': i.id,
                'type': i.incident_type,
                'status': i.status
            }
            for i in Incident.objects.order_by('-detected_at')[:10]
        ]
    })


@csrf_exempt
def resolve_incident_api(request, incident_id):
    """Mark an incident as resolved and update its status to 'completed'.

    Called when an operator reviews an incident and confirms it has been
    handled. Sets resolved=True and status='completed' on the Incident record.
    Returns 404 if the incident does not exist.
    """
    try:
        incident = Incident.objects.get(id=incident_id)
        incident.resolved = True
        incident.status = 'completed'
        incident.save()
        return JsonResponse({'success': True})
    except Incident.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Incident not found'}, status=404)


@csrf_exempt
def check_incident_status_api(request, log_id):
    """Return the processing status of an incident detection job (polled by the frontend).

    The 'incident_processing' waiting-room template calls this endpoint on an interval.
    Returns { is_processed: bool, log_id: int }.
    Returns 404 with an error message if the IncidentLog does not exist.
    """
    try:
        log = IncidentLog.objects.get(id=log_id)
        return JsonResponse({
            'is_processed': log.is_processed,
            'log_id': log_id
        })
    except IncidentLog.DoesNotExist:
        return JsonResponse({
            'is_processed': False,
            'error': 'Incident log not found'
        }, status=404)