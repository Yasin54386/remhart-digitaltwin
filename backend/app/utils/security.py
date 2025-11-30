"""
REMHART Digital Twin - Security Utilities
==========================================
Password hashing, JWT token generation, and authentication utilities.

Security Features:
- Bcrypt password hashing
- JWT token authentication
- Role-based access control (RBAC)

Author: REMHART Team
Date: 2025
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "d9FA4gH8kL2pQ7wX1zR6vT0bY3mN5cJq")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context using bcrypt
# Bcrypt is recommended for password hashing due to its adaptive nature
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Optional OAuth2 scheme (doesn't raise error if token is missing)
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)


# ============================================
# Password Hashing Functions
# ============================================

def hash_password(password: str) -> str:
    # Encode password to bytes
    password_bytes = password.encode("utf-8")
    # Truncate to 72 bytes safely
    if len(password_bytes) > 72:
        password = password_bytes[:72].decode("utf-8", "ignore")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        plain_password = password_bytes[:72].decode("utf-8", "ignore")
    return pwd_context.verify(plain_password, hashed_password)


# ============================================
# JWT Token Functions
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of data to encode in token (e.g., {"sub": "username"})
        expires_delta: Optional expiration time delta
        
    Returns:
        str: Encoded JWT token
        
    Example:
        token = create_access_token({"sub": "admin", "role": "admin"})
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Encode token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
        
    Example:
        payload = verify_token(token)
        username = payload.get("sub")
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================
# User Authentication Dependencies
# ============================================

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user from JWT token.
    Use this as a dependency in protected routes.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        dict: User information from token
        
    Raises:
        HTTPException: If token is invalid
        
    Usage:
        @app.get("/protected")
        async def protected_route(current_user = Depends(get_current_user)):
            return {"user": current_user}
    """
    payload = verify_token(token)
    username: str = payload.get("sub")
    
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    return payload


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme_optional)):
    """
    Get current user if authenticated, otherwise return None.
    Use this for endpoints that work with or without authentication.

    Args:
        token: Optional JWT token from Authorization header

    Returns:
        dict | None: User information from token, or None if not authenticated

    Usage:
        @app.get("/public-or-private")
        async def flexible_route(current_user = Depends(get_optional_user)):
            if current_user:
                return {"message": f"Hello {current_user['sub']}"}
            return {"message": "Hello anonymous"}
    """
    if not token:
        return None

    try:
        payload = verify_token(token)
        return payload
    except Exception:
        # Return None for any authentication errors (invalid token, expired, etc.)
        return None


def check_role(required_role: str):
    """
    Check if current user has required role.
    Role hierarchy: admin > operator > analyst > viewer
    
    Args:
        required_role: Minimum required role
        
    Returns:
        Function: Dependency function for role checking
        
    Usage:
        @app.get("/admin-only")
        async def admin_route(user = Depends(check_role("admin"))):
            return {"message": "Admin access granted"}
    """
    async def role_checker(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        # Role hierarchy
        role_hierarchy = {
            "viewer": 1,
            "analyst": 2,
            "operator": 3,
            "admin": 4
        }
        
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


# ============================================
# Helper Functions
# ============================================

def authenticate_user(db: Session, username: str, password: str):
    """
    Authenticate a user with username and password.
    
    Args:
        db: Database session
        username: Username
        password: Plain text password
        
    Returns:
        User object if authentication successful, None otherwise
        
    Example:
        from app.models.db_models import User
        user = authenticate_user(db, "admin", "password123")
    """
    from app.models.db_models import User
    
    # Find user by username
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        return None
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        return None
    
    # Check if user is active
    if not user.is_active:
        return None
    
    return user


def create_default_users(db: Session):
    """
    Create default users for initial setup.
    Only creates users if database is empty.
    
    Default users:
    - admin/admin123 (Admin role)
    - operator/operator123 (Operator role)
    - analyst/analyst123 (Analyst role)
    - viewer/viewer123 (Viewer role)
    
    Args:
        db: Database session
        
    Example:
        create_default_users(db)
    """
    from app.models.db_models import User
    
    # Check if users already exist
    if db.query(User).first():
        print("✅ Users already exist in database")
        return
    
    # Default users
    default_users = [
        {
            "username": "admin",
            "email": "admin@remhart.cdu.edu.au",
            "password": "admin123",
            "full_name": "System Administrator",
            "role": "admin"
        },
        {
            "username": "operator",
            "email": "operator@remhart.cdu.edu.au",
            "password": "operator123",
            "full_name": "Grid Operator",
            "role": "operator"
        },
        {
            "username": "analyst",
            "email": "analyst@remhart.cdu.edu.au",
            "password": "analyst123",
            "full_name": "Data Analyst",
            "role": "analyst"
        },
        {
            "username": "viewer",
            "email": "viewer@remhart.cdu.edu.au",
            "password": "viewer123",
            "full_name": "System Viewer",
            "role": "viewer"
        }
    ]
    
    # Create users
    for user_data in default_users:
        user = User(
            username=user_data["username"],
            email=user_data["email"],
            hashed_password=hash_password(user_data["password"]),
            full_name=user_data["full_name"],
            role=user_data["role"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        db.add(user)
    
    db.commit()
    print("Default users created successfully")
    print("\nDefault Login Credentials:")
    print("=" * 50)
    for user_data in default_users:
        print(f"Username: {user_data['username']:12} | Password: {user_data['password']:15} | Role: {user_data['role']}")
    print("=" * 50)