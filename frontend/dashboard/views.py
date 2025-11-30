"""
REMHART Digital Twin - Django Views
====================================
Handles HTTP requests and renders templates.

Author: REMHART Team
Date: 2025
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
import httpx
import asyncio

FASTAPI_BASE_URL = settings.FASTAPI_BASE_URL
FASTAPI_WS_URL = settings.FASTAPI_WS_URL


def landing_page(request):
    """
    Landing page with REMHART information and login button.
    """
    return render(request, 'landing.html')


def login_view(request):
    """
    User login page and authentication handling.
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Call FastAPI login endpoint
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{FASTAPI_BASE_URL}/api/auth/login",
                    json={"username": username, "password": password}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Store token in session
                    request.session['access_token'] = data['access_token']
                    request.session['user'] = data['user']
                    return redirect('dashboard:dashboard')
                else:
                    error_message = "Invalid username or password"
                    return render(request, 'login.html', {'error': error_message})
                    
        except Exception as e:
            error_message = f"Connection error: {str(e)}"
            return render(request, 'login.html', {'error': error_message})
    
    return render(request, 'login.html')


def logout_view(request):
    """
    User logout - clear session and redirect to landing page.
    """
    request.session.flush()
    return redirect('dashboard:landing')


def require_login(view_func):
    """
    Decorator to require authentication for views.
    """
    def wrapper(request, *args, **kwargs):
        if 'access_token' not in request.session:
            return redirect('dashboard:login')
        return view_func(request, *args, **kwargs)
    return wrapper


@require_login
def dashboard(request):
    """
    Main dashboard view with overview metrics.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'dashboard.html', context)


@require_login
def predictive_maintenance(request):
    """
    Predictive maintenance visualization page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'graphs/predictive_maintenance.html', context)


@require_login
def energy_flow(request):
    """
    Energy flow overview visualization page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'graphs/energy_flow.html', context)


@require_login
def realtime_monitoring(request):
    """
    Real-time grid monitoring visualization page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'graphs/realtime_monitoring.html', context)


@require_login
def decision_making(request):
    """
    Data-driven decision making visualization page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'graphs/decision_making.html', context)


@require_login
def get_latest_data_ajax(request):
    """
    AJAX endpoint to get latest grid data.
    """
    token = request.session.get('access_token')
    
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{FASTAPI_BASE_URL}/api/grid/data/latest",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                return JsonResponse(response.json())
            else:
                return JsonResponse({"error": "Failed to fetch data"}, status=500)
                
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
@require_login
def simulator(request):
    """
    Grid simulator page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'simulator.html', context)

@require_login
def reports(request):
    """
    Reports and data export page.
    """
    context = {
        'user': request.session.get('user', {}),
        'ws_url': FASTAPI_WS_URL,
        'api_url': FASTAPI_BASE_URL
    }
    return render(request, 'reports.html', context)