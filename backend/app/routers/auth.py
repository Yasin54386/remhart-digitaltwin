"""
REMHART Digital Twin - Authentication Router
============================================
Handles user authentication, login, and token management.

Endpoints:
- POST /api/auth/login - User login
- GET /api/auth/me - Get current user info
- POST /api/auth/refresh - Refresh access token

Author: REMHART Team
Date: 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.database import get_db
from app.models.db_models import User
from app.schemas.grid_data import UserLogin, Token, UserResponse
from app.utils.security import (
    authenticate_user,
    create_access_token,
    get_current_user,
    hash_password
)

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    User login endpoint.
    
    Accepts username and password, returns JWT access token.
    
    Args:
        user_credentials: Username and password
        db: Database session
        
    Returns:
        Token object with access_token and user info
        
    Raises:
        HTTPException: If credentials are invalid
        
    Example:
        POST /api/auth/login
        {
            "username": "admin",
            "password": "admin123"
        }
    """
    # Authenticate user
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role
        }
    )
    
    # Update last login time
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Return token and user info
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Requires valid JWT token in Authorization header.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        User information
        
    Example:
        GET /api/auth/me
        Authorization: Bearer <token>
    """
    return {
        "id": current_user.get("user_id"),
        "username": current_user.get("sub"),
        "email": current_user.get("email", ""),
        "full_name": current_user.get("full_name", ""),
        "role": current_user.get("role"),
        "is_active": True
    }


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Refresh access token.
    
    Generates a new access token for the current user.
    
    Args:
        current_user: Current user from JWT token
        db: Database session
        
    Returns:
        New token
        
    Example:
        POST /api/auth/refresh
        Authorization: Bearer <old_token>
    """
    # Get user from database
    user = db.query(User).filter(User.username == current_user.get("sub")).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role
        }
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout endpoint (placeholder).
    
    In JWT-based auth, logout is handled client-side by discarding the token.
    This endpoint exists for API consistency.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        Success message
    """
    return {
        "message": "Successfully logged out",
        "detail": "Please discard your access token on the client side"
    }