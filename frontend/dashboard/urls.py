"""
REMHART Digital Twin - Dashboard URL Configuration
"""

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Landing page
    path('', views.landing_page, name='landing'),
    
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Simulator
    path('simulator/', views.simulator, name='simulator'),

    # Reports
    path('reports/', views.reports, name='reports'),

    # Graph visualization pages
    path('predictive-maintenance/', views.predictive_maintenance, name='predictive_maintenance'),
    path('energy-flow/', views.energy_flow, name='energy_flow'),
    path('realtime-monitoring/', views.realtime_monitoring, name='realtime_monitoring'),
    path('decision-making/', views.decision_making, name='decision_making'),
    
    # API endpoints for AJAX
    path('api/latest-data/', views.get_latest_data_ajax, name='api_latest_data'),
]