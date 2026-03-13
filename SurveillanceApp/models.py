# SurveillanceApp\models.py
from django.db import models

class Incident(models.Model):
    INCIDENT_TYPES = [('accident', 'Vehicle Accident'), ('fight', 'Fight Detected')]
    STATUS_CHOICES = [('new', 'New'), ('reviewed', 'Reviewed'), ('processing', 'Processing'), ('completed', 'Completed')]

    incident_type = models.CharField(max_length=20, choices=INCIDENT_TYPES)
    location = models.CharField(max_length=255, default="Main Feed")
    timestamp = models.DateTimeField(auto_now_add=True)
    detected_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    snapshot = models.ImageField(upload_to='incidents/', null=True, blank=True)
    resolved = models.BooleanField(default=False)

    def __str__(self):
        return f"#{self.id} {self.incident_type}"

class SearchLog(models.Model):
    license_plate = models.CharField(max_length=20)
    searched_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)

    def __str__(self):
        return f"Search: {self.license_plate}"

class IncidentLog(models.Model):
    """Tracks async anomaly detection requests sent to Kaggle"""
    video_path = models.CharField(max_length=500, default="")
    incident = models.ForeignKey(Incident, on_delete=models.SET_NULL, null=True, blank=True)
    is_processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"IncidentLog #{self.id} - {'Done' if self.is_processed else 'Processing'}"

class Vehicle(models.Model):
    license_plate = models.CharField(max_length=20, db_index=True)
    # Linked to either an Incident (auto-detection) or a User Search (manual)
    incident = models.ForeignKey(Incident, on_delete=models.CASCADE, related_name='vehicles', null=True, blank=True)
    search_log = models.ForeignKey(SearchLog, on_delete=models.CASCADE, related_name='vehicles', null=True, blank=True)
    
    detection_confidence = models.FloatField(default=0.0)

    def __str__(self):
        return self.license_plate

class VehicleDetection(models.Model):
    """Stores where a vehicle was seen. Camera info is now stored as strings."""
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name='detections')
    
    # HARDCODED CAMERA INFO STORED HERE
    camera_name = models.CharField(max_length=100,default="cam1")    
    camera_location = models.CharField(max_length=255,default="Main Street Intersection")  
    
    timestamp = models.CharField(max_length=50)         
    matched_text = models.CharField(max_length=20)
    snapshot = models.ImageField(upload_to='detections/', null=True, blank=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['detected_at']