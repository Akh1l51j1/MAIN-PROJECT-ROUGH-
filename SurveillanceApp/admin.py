# surveillance/admin.py

from django.contrib import admin
from .models import Incident, Vehicle, SearchLog, VehicleDetection

@admin.register(Incident)
class IncidentAdmin(admin.ModelAdmin):
    list_display = ('id', 'incident_type', 'status', 'detected_at', 'resolved')
    list_display_links = ('id', 'incident_type')
    list_filter = ('incident_type', 'status', 'resolved')


@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ('license_plate', 'incident', 'search_log')

@admin.register(VehicleDetection)
class VehicleDetectionAdmin(admin.ModelAdmin):
    list_display = ('vehicle', 'camera_name', 'timestamp', 'matched_text')

@admin.register(SearchLog)
class SearchLogAdmin(admin.ModelAdmin):
    list_display = ('license_plate', 'searched_at', 'is_processed')