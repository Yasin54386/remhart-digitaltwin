"""
REMHART Digital Twin - FastAPI Main Application
===============================================
Main entry point for the FastAPI backend server.

Features:
- RESTful API endpoints for grid data
- WebSocket support for real-time updates
- JWT authentication
- Role-based access control
- CORS enabled for Django frontend

Author: REMHART Team
Date: 2025
Version: 1.0.0
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uvicorn
from contextlib import asynccontextmanager

# Import database and models
from app.database import init_db, check_db_connection, get_db
from app.utils.security import create_default_users

# Import routers
from app.routers import auth, grid_data, websocket_router, simulator

# ============================================
# Application Lifecycle Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.
    
    Startup:
    - Check database connection
    - Initialize database tables
    - Create default users
    
    Shutdown:
    - Cleanup resources
    """
    # Startup
    print("\n" + "="*60)
    print("Starting REMHART Digital Twin Backend")
    print("="*60)
    
    # Check database connection
    if not check_db_connection():
        print("⚠️  Database connection failed. Please check your configuration.")
    
    # Initialize database
    init_db()
    
    # Create default users
    db = next(get_db())
    create_default_users(db)
    db.close()
    
    print("\nBackend server started successfully!")
    print(f"API Documentation: http://localhost:8001/docs")
    print(f"WebSocket: ws://localhost:8001/ws/grid-data")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down REMHART Digital Twin Backend...")


# ============================================
# FastAPI Application Instance
# ============================================

app = FastAPI(
    title="REMHART Digital Twin API",
    description="""
    ## Smart Grid Data Management System
    
    RESTful API for managing and monitoring smart grid data from REMHART.
    
    ### Features:
    - Real-time grid data monitoring
    - JWT-based authentication
    - AI/ML integration for predictive analytics
    - WebSocket support for live updates
    - Time-series data visualization
    
    ### Authentication:
    Use `/api/auth/login` to obtain an access token.
    Then include it in the Authorization header: `Bearer <token>`
    
    ### Developed By:
    Charles Darwin University - Energy and Resources Institute (ERI)
    REMHART Project Team
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================
# CORS Middleware Configuration
# ============================================

# Allow Django frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",      # Django dev server
        "http://127.0.0.1:8000",
        "http://localhost:3000",      # Alternative frontend ports
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ============================================
# Include Routers
# ============================================

# Authentication routes
app.include_router(
    auth.router,
    prefix="/api/auth",
    tags=["Authentication"]
)

# Grid data routes
app.include_router(
    grid_data.router,
    prefix="/api/grid",
    tags=["Grid Data"]
)

# WebSocket routes
app.include_router(
    websocket_router.router,
    prefix="/ws",
    tags=["WebSocket"]
)

# Simulator routes
app.include_router(
    simulator.router,
    prefix="/api/simulator",
    tags=["Simulator"]
)


# ============================================
# Root Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint - provides basic information.
    """
    return {
        "message": "REMHART Digital Twin API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "authentication": "/api/auth/login",
            "grid_data": "/api/grid/data",
            "websocket": "ws://localhost:8001/ws/grid-data"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint - verify API and database status.
    """
    try:
        # Test database query
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": "2025-01-15T10:30:00Z"
    }


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with custom message"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors with custom message"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support."
        }
    )


# ============================================
# Run Server (for development)
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )