"""
Enterprise Features
User management, API endpoints, authentication, and enterprise-grade functionality
"""

import streamlit as st
import streamlit_authenticator as stauth
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import jwt
import bcrypt
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import asyncio
import threading
from pydantic import BaseModel
import redis
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default="user")  # user, admin, enterprise
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    api_key = Column(String, unique=True)
    usage_limit = Column(Integer, default=1000)  # queries per month
    usage_count = Column(Integer, default=0)

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    user_id = Column(Integer)
    name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    rate_limit = Column(Integer, default=100)  # requests per hour

class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    domain = Column(String)
    plan = Column(String, default="basic")  # basic, professional, enterprise
    max_users = Column(Integer, default=10)
    max_documents = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    role: str = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str
    model: str = "gpt-4-turbo-preview"
    max_tokens: int = 1000
    temperature: float = 0.7

class DocumentUpload(BaseModel):
    filename: str
    content: str  # base64 encoded
    processing_options: Dict[str, Any] = {}

# Database setup
DATABASE_URL = "sqlite:///./enterprise.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserManager:
    """User management system"""
    
    def __init__(self):
        Base.metadata.create_all(bind=engine)
        self.secret_key = "your-secret-key-here"  # Should be from environment
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, user_data: UserCreate, db: Session) -> User:
        """Create new user"""
        # Check if user exists
        existing_user = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user
        hashed_password = self.hash_password(user_data.password)
        api_key = self.generate_api_key()
        
        db_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=user_data.role,
            api_key=api_key
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return db_user
    
    def authenticate_user(self, username: str, password: str, db: Session) -> Optional[User]:
        """Authenticate user"""
        user = db.query(User).filter(User.username == username).first()
        
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    
    def generate_api_key(self) -> str:
        """Generate API key"""
        import secrets
        return f"fpa_{secrets.token_urlsafe(32)}"
    
    def create_jwt_token(self, user: User) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

class RateLimiter:
    """Rate limiting system using Redis"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except:
            logger.warning("Redis not available, using in-memory rate limiting")
            self.redis_client = None
            self.memory_store = {}
    
    def is_allowed(self, key: str, limit: int, window: int = 3600) -> bool:
        """Check if request is allowed within rate limit"""
        if self.redis_client:
            try:
                current = self.redis_client.get(key)
                if current is None:
                    self.redis_client.setex(key, window, 1)
                    return True
                elif int(current) < limit:
                    self.redis_client.incr(key)
                    return True
                else:
                    return False
            except:
                # Fallback to memory-based limiting
                pass
        
        # Memory-based fallback
        now = datetime.utcnow()
        if key not in self.memory_store:
            self.memory_store[key] = {"count": 1, "reset_time": now + timedelta(seconds=window)}
            return True
        
        if now > self.memory_store[key]["reset_time"]:
            self.memory_store[key] = {"count": 1, "reset_time": now + timedelta(seconds=window)}
            return True
        
        if self.memory_store[key]["count"] < limit:
            self.memory_store[key]["count"] += 1
            return True
        
        return False

# FastAPI application
app = FastAPI(title="Fenestration Pro AI API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
user_manager = UserManager()
rate_limiter = RateLimiter()
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security), 
                    db: Session = Depends(get_db)) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = user_manager.verify_jwt_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user

def check_rate_limit(user: User):
    """Check rate limit for user"""
    if not rate_limiter.is_allowed(f"user_{user.id}", user.usage_limit if user.role != "enterprise" else 10000):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# API Endpoints
@app.post("/api/auth/register")
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    try:
        user = user_manager.create_user(user_data, db)
        token = user_manager.create_jwt_token(user)
        
        return {
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "api_key": user.api_key
            },
            "token": token
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """User login"""
    user = user_manager.authenticate_user(login_data.username, login_data.password, db)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = user_manager.create_jwt_token(user)
    
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        },
        "token": token
    }

@app.get("/api/user/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get user profile"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "usage_count": current_user.usage_count,
        "usage_limit": current_user.usage_limit,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

@app.post("/api/query")
async def query_documents(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Query documents via API"""
    check_rate_limit(current_user)
    
    try:
        # Import RAG pipeline
        from advanced_rag import get_rag_pipeline
        
        # Get API key from environment or user settings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Initialize RAG pipeline
        pipeline = get_rag_pipeline(api_key)
        
        # Process query
        start_time = datetime.utcnow()
        result = await pipeline.query_async(query_request.query)
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update usage count
        current_user.usage_count += 1
        db.commit()
        
        # Track analytics
        from analytics_dashboard import track_query
        track_query(
            query_request.query,
            response_time,
            query_request.model,
            tokens_used=len(result.get('answer', '').split()) * 1.3,  # Rough estimate
            error_occurred=False
        )
        
        return {
            "answer": result.get('answer', ''),
            "source_documents": result.get('source_documents', []),
            "response_time": response_time,
            "model_used": query_request.model,
            "usage_remaining": current_user.usage_limit - current_user.usage_count
        }
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    document: DocumentUpload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and process document via API"""
    check_rate_limit(current_user)
    
    try:
        import base64
        import tempfile
        from pathlib import Path
        
        # Decode base64 content
        file_content = base64.b64decode(document.content)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(document.filename).suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            from advanced_rag import get_rag_pipeline
            
            api_key = os.getenv("OPENAI_API_KEY")
            pipeline = get_rag_pipeline(api_key)
            
            start_time = datetime.utcnow()
            results = await pipeline.process_documents_async([tmp_file_path], document.processing_options)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Track document processing
            from analytics_dashboard import track_document_processing
            track_document_processing(
                document.filename,
                len(file_content),
                processing_time,
                results.get('total_chunks', 0),
                len(results.get('errors', [])) == 0
            )
            
            return {
                "filename": document.filename,
                "processing_time": processing_time,
                "chunks_created": results.get('total_chunks', 0),
                "success": len(results.get('errors', [])) == 0,
                "errors": results.get('errors', [])
            }
            
        finally:
            # Cleanup temp file
            Path(tmp_file_path).unlink()
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/dashboard")
async def get_analytics(current_user: User = Depends(get_current_user)):
    """Get analytics data"""
    if current_user.role not in ["admin", "enterprise"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        from analytics_dashboard import AnalyticsDatabase
        
        db = AnalyticsDatabase()
        session_data = db.get_session_stats(30)
        query_data = db.get_query_stats(30)
        
        return {
            "sessions": session_data.to_dict('records') if not session_data.empty else [],
            "queries": query_data.to_dict('records') if not query_data.empty else [],
            "summary": {
                "total_sessions": len(session_data),
                "total_queries": len(query_data),
                "avg_response_time": query_data['response_time'].mean() if not query_data.empty else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit Authentication Integration
class StreamlitAuth:
    """Streamlit authentication integration"""
    
    def __init__(self):
        self.user_manager = UserManager()
        
    def setup_authentication(self):
        """Setup Streamlit authentication"""
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        
        # Authentication UI
        if not st.session_state.authenticated:
            self.render_login_page()
        else:
            self.render_user_info()
    
    def render_login_page(self):
        """Render login/registration page"""
        st.title("🔐 Authentication")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit and username and password:
                    db = SessionLocal()
                    try:
                        user = self.user_manager.authenticate_user(username, password, db)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    finally:
                        db.close()
        
        with tab2:
            st.subheader("Register")
            
            with st.form("register_form"):
                username = st.text_input("Username", key="reg_username")
                email = st.text_input("Email", key="reg_email")
                full_name = st.text_input("Full Name", key="reg_full_name")
                password = st.text_input("Password", type="password", key="reg_password")
                role = st.selectbox("Role", ["user", "admin"], key="reg_role")
                submit = st.form_submit_button("Register")
                
                if submit and all([username, email, full_name, password]):
                    db = SessionLocal()
                    try:
                        user_data = UserCreate(
                            username=username,
                            email=email,
                            full_name=full_name,
                            password=password,
                            role=role
                        )
                        user = self.user_manager.create_user(user_data, db)
                        st.success("Registration successful! Please login.")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
                    finally:
                        db.close()
    
    def render_user_info(self):
        """Render user information in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("👤 User Profile")
            
            user = st.session_state.user
            st.write(f"**Name:** {user.full_name}")
            st.write(f"**Role:** {user.role.title()}")
            st.write(f"**Usage:** {user.usage_count}/{user.usage_limit}")
            
            # Usage progress bar
            usage_percent = (user.usage_count / user.usage_limit) * 100
            st.progress(usage_percent / 100)
            st.caption(f"{usage_percent:.1f}% of monthly limit used")
            
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.user = None
                st.rerun()
            
            # API Key display
            with st.expander("🔑 API Key"):
                st.code(user.api_key)
                st.caption("Use this key for API access")

def run_api_server():
    """Run FastAPI server in background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def start_api_server():
    """Start API server in background thread"""
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    return api_thread

# Export main classes and functions
__all__ = [
    'UserManager',
    'StreamlitAuth',
    'RateLimiter',
    'app',
    'start_api_server',
    'User',
    'APIKey',
    'Organization'
]