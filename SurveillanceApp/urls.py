# SurveillanceApp\urls.py
from django.urls import path
from . import views, api_views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('', views.dashboard, name='dashboard'),

    path('anomaly-detection/', views.anomaly_detection, name='anomaly_detection'),
    path('fight-anomaly-detection/', views.fight_anomaly_detection, name='fight_anomaly_detection'),
    path('anomaly-detection/processing/<int:log_id>/', views.incident_processing, name='incident_processing'),

    path('incident/<int:incident_id>/', views.incident_detail, name='incident_detail'),

    path('tracking/', views.tracking, name='tracking'),
    path('tracking/processing/<int:search_id>/', views.processing, name='processing'),
    path('tracking/<int:search_id>/', views.tracking_results, name='tracking_results'),

    # APIs
    path('api/incidents/', api_views.get_incidents_api, name='get_incidents_api'),
    path('api/incident/create/', api_views.create_incident_api, name='create_incident_api'),
    path('api/incident/<int:incident_id>/resolve/', api_views.resolve_incident_api),
    path('api/search/result/', api_views.search_result_api, name='search_result_api'),
    path('api/check-status/<int:search_id>/', api_views.check_status_api, name='check_status_api'),
    path('api/check-incident-status/<int:log_id>/', api_views.check_incident_status_api, name='check_incident_status_api'),
]