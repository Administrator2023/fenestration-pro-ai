"""
Utility functions for Fenestration Pro AI - SOTA Edition
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Advanced document processing utilities"""
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def extract_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from uploaded file"""
        path = Path(file_path)
        return {
            "filename": path.name,
            "size": path.stat().st_size,
            "extension": path.suffix,
            "created": datetime.now().isoformat()
        }
    
    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
        """Validate if file type is allowed"""
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in allowed_types

class AnalyticsManager:
    """Manage analytics and statistics"""
    
    @staticmethod
    def calculate_session_stats() -> Dict[str, Any]:
        """Calculate current session statistics"""
        messages = st.session_state.get('messages', [])
        
        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']
        
        # Calculate response times
        response_times = [
            m.get('response_time', 0) for m in assistant_messages 
            if 'response_time' in m
        ]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_messages": len(messages),
            "user_queries": len(user_messages),
            "assistant_responses": len(assistant_messages),
            "avg_response_time": avg_response_time,
            "total_response_time": sum(response_times),
            "session_duration": AnalyticsManager.get_session_duration()
        }
    
    @staticmethod
    def get_session_duration() -> float:
        """Get current session duration in minutes"""
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        duration = datetime.now() - st.session_state.session_start
        return duration.total_seconds() / 60
    
    @staticmethod
    def export_analytics(format_type: str = "json") -> str:
        """Export analytics data in specified format"""
        stats = AnalyticsManager.calculate_session_stats()
        messages = st.session_state.get('messages', [])
        
        export_data = {
            "session_stats": stats,
            "messages": messages,
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format_type == "csv":
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(messages)
            return df.to_csv(index=False)
        else:
            return json.dumps(export_data, indent=2, default=str)

class UIHelpers:
    """UI helper functions"""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
        """Create a styled metric card"""
        delta_html = f'<p style="color: #10b981; font-size: 0.9rem;">{delta}</p>' if delta else ""
        
        return f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #64748b;">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: #1e293b;">{value}</h2>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_status_badge(status: str, text: str) -> str:
        """Create a status badge"""
        colors = {
            "success": "#10b981",
            "warning": "#f59e0b", 
            "error": "#ef4444",
            "info": "#3b82f6"
        }
        
        color = colors.get(status, "#64748b")
        
        return f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 500;
        ">{text}</span>
        """
    
    @staticmethod
    def create_progress_bar(progress: float, label: str = "") -> str:
        """Create a custom progress bar"""
        return f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>{label}</span>
                <span>{progress:.1%}</span>
            </div>
            <div style="
                width: 100%;
                height: 8px;
                background-color: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress * 100}%;
                    height: 100%;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """

class ErrorHandler:
    """Enhanced error handling and logging"""
    
    @staticmethod
    def log_error(error: Exception, context: str = "") -> None:
        """Log error with context"""
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    @staticmethod
    def display_error(error: Exception, user_friendly: bool = True) -> None:
        """Display error to user"""
        if user_friendly:
            st.error("An error occurred. Please try again or contact support.")
        else:
            st.error(f"Error: {str(error)}")
    
    @staticmethod
    def handle_api_error(error: Exception) -> str:
        """Handle API-specific errors"""
        error_str = str(error).lower()
        
        if "api key" in error_str:
            return "Invalid or missing API key. Please check your OpenAI API key."
        elif "rate limit" in error_str:
            return "Rate limit exceeded. Please wait a moment and try again."
        elif "quota" in error_str:
            return "API quota exceeded. Please check your OpenAI account."
        elif "timeout" in error_str:
            return "Request timed out. Please try again."
        else:
            return f"API Error: {str(error)}"

class CacheManager:
    """Manage caching for improved performance"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def cache_embeddings(text_chunks: List[str], model: str) -> List[List[float]]:
        """Cache embeddings to avoid recomputation"""
        # This would integrate with your embedding model
        pass
    
    @staticmethod
    @st.cache_resource
    def load_sentence_transformer(model_name: str):
        """Cache sentence transformer model"""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    
    @staticmethod
    def clear_cache():
        """Clear all caches"""
        st.cache_data.clear()
        st.cache_resource.clear()

class SecurityUtils:
    """Security utilities"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security"""
        # Remove path traversal attempts
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        return filename
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 200) -> bool:
        """Validate file size"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """Create a download link for data"""
    import base64
    
    b64_data = base64.b64encode(data.encode()).decode()
    
    return f"""
    <a href="data:{mime_type};base64,{b64_data}" 
       download="{filename}"
       style="
           display: inline-block;
           padding: 0.5rem 1rem;
           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
           color: white;
           text-decoration: none;
           border-radius: 0.5rem;
           font-weight: 500;
       ">
       📥 Download {filename}
    </a>
    """