"""
Authentication and Authorization for Ocean Data Integration Platform
JWT token management, user authentication, and role-based access control
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

class AuthManager:
    """Manages authentication and authorization for the Ocean Data Platform."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """Get the current user from JWT token."""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # In production, fetch user from database
        # For demo purposes, return mock user data
        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "role": payload.get("role", "user"),
            "organization": payload.get("organization", "Unknown")
        }
    
    def require_role(self, required_role: str):
        """Create a dependency that requires a specific role."""
        def role_checker(current_user: Dict[str, Any] = Depends(self.get_current_user)) -> Dict[str, Any]:
            if current_user.get("role") != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return current_user
        return role_checker
    
    def require_admin(self):
        """Require admin role."""
        return self.require_role("admin")
    
    def require_researcher(self):
        """Require researcher role or higher."""
        def researcher_checker(current_user: Dict[str, Any] = Depends(self.get_current_user)) -> Dict[str, Any]:
            allowed_roles = ["admin", "researcher"]
            if current_user.get("role") not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Researcher or admin role required"
                )
            return current_user
        return researcher_checker

# Global auth manager instance
auth_manager = AuthManager()

# Pydantic models for authentication
from pydantic import BaseModel, EmailStr

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    organization: str
    role: str = "user"

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class UserProfile(BaseModel):
    user_id: int
    email: str
    name: str
    organization: str
    role: str
    is_active: bool
    created_at: datetime

# Authentication endpoints
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return JWT token."""
    try:
        # In production, verify credentials against database
        # For demo purposes, accept any email/password combination
        if not user_credentials.email or not user_credentials.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Mock user data - in production, fetch from database
        user_data = {
            "sub": "1",  # User ID
            "email": user_credentials.email,
            "role": "researcher",
            "organization": "Ocean Research Institute"
        }
        
        # Create access token
        access_token = auth_manager.create_access_token(data=user_data)
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=auth_manager.access_token_expire_minutes * 60
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

@auth_router.post("/register", response_model=UserProfile)
async def register(user_data: UserRegister):
    """Register a new user."""
    try:
        # In production, hash password and store in database
        hashed_password = auth_manager.get_password_hash(user_data.password)
        
        # Mock user creation - in production, save to database
        new_user = {
            "user_id": 1,  # Mock ID
            "email": user_data.email,
            "name": user_data.name,
            "organization": user_data.organization,
            "role": user_data.role,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"User registered: {user_data.email}")
        return UserProfile(**new_user)
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed"
        )

@auth_router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(auth_manager.get_current_user)):
    """Get current user profile."""
    try:
        # Mock user profile - in production, fetch from database
        user_profile = {
            "user_id": int(current_user["user_id"]),
            "email": current_user["email"],
            "name": "Demo User",
            "organization": current_user.get("organization", "Unknown"),
            "role": current_user["role"],
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        return UserProfile(**user_profile)
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )

@auth_router.post("/refresh")
async def refresh_token(current_user: Dict[str, Any] = Depends(auth_manager.get_current_user)):
    """Refresh JWT token."""
    try:
        # Create new token with same user data
        user_data = {
            "sub": current_user["user_id"],
            "email": current_user["email"],
            "role": current_user["role"],
            "organization": current_user.get("organization", "Unknown")
        }
        
        new_token = auth_manager.create_access_token(data=user_data)
        
        return Token(
            access_token=new_token,
            token_type="bearer",
            expires_in=auth_manager.access_token_expire_minutes * 60
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

# Role-based access control decorators
def require_admin():
    """Require admin role."""
    return auth_manager.require_admin()

def require_researcher():
    """Require researcher role or higher."""
    return auth_manager.require_researcher()

def require_user():
    """Require any authenticated user."""
    return auth_manager.get_current_user

# Permission checking functions
def can_create_species_data(user: Dict[str, Any]) -> bool:
    """Check if user can create species data."""
    allowed_roles = ["admin", "researcher", "fisherman"]
    return user.get("role") in allowed_roles

def can_create_vessel_data(user: Dict[str, Any]) -> bool:
    """Check if user can create vessel data."""
    allowed_roles = ["admin", "fisherman", "vessel_operator"]
    return user.get("role") in allowed_roles

def can_create_edna_data(user: Dict[str, Any]) -> bool:
    """Check if user can create eDNA data."""
    allowed_roles = ["admin", "researcher", "scientist"]
    return user.get("role") in allowed_roles

def can_access_analytics(user: Dict[str, Any]) -> bool:
    """Check if user can access analytics."""
    allowed_roles = ["admin", "researcher", "policy_maker"]
    return user.get("role") in allowed_roles

def can_make_predictions(user: Dict[str, Any]) -> bool:
    """Check if user can make AI predictions."""
    allowed_roles = ["admin", "researcher", "scientist", "fisherman"]
    return user.get("role") in allowed_roles
