"""
Advanced Analytics Dashboard
Comprehensive monitoring and analytics system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psutil
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionMetrics:
    """Session-level metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_queries: int
    total_documents: int
    total_chunks: int
    avg_response_time: float
    total_tokens_used: int
    user_satisfaction: Optional[float]
    errors_count: int

@dataclass
class QueryMetrics:
    """Individual query metrics"""
    query_id: str
    session_id: str
    timestamp: datetime
    query_text: str
    response_time: float
    tokens_used: int
    model_used: str
    retrieval_score: float
    user_rating: Optional[int]
    error_occurred: bool

@dataclass
class DocumentMetrics:
    """Document processing metrics"""
    document_id: str
    filename: str
    file_size: int
    processing_time: float
    chunk_count: int
    embedding_time: float
    success: bool
    error_message: Optional[str]

class AnalyticsDatabase:
    """SQLite database for analytics storage"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_queries INTEGER DEFAULT 0,
                total_documents INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                total_tokens_used INTEGER DEFAULT 0,
                user_satisfaction REAL,
                errors_count INTEGER DEFAULT 0
            )
        """)
        
        # Queries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TIMESTAMP,
                query_text TEXT,
                response_time REAL,
                tokens_used INTEGER,
                model_used TEXT,
                retrieval_score REAL,
                user_rating INTEGER,
                error_occurred BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT,
                file_size INTEGER,
                processing_time REAL,
                chunk_count INTEGER,
                embedding_time REAL,
                success BOOLEAN,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                active_sessions INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_session(self, session: SessionMetrics):
        """Insert session metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, start_time, end_time, total_queries, total_documents, 
             total_chunks, avg_response_time, total_tokens_used, user_satisfaction, errors_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id, session.start_time, session.end_time,
            session.total_queries, session.total_documents, session.total_chunks,
            session.avg_response_time, session.total_tokens_used, 
            session.user_satisfaction, session.errors_count
        ))
        
        conn.commit()
        conn.close()
    
    def insert_query(self, query: QueryMetrics):
        """Insert query metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO queries 
            (query_id, session_id, timestamp, query_text, response_time, 
             tokens_used, model_used, retrieval_score, user_rating, error_occurred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query.query_id, query.session_id, query.timestamp, query.query_text,
            query.response_time, query.tokens_used, query.model_used,
            query.retrieval_score, query.user_rating, query.error_occurred
        ))
        
        conn.commit()
        conn.close()
    
    def insert_document(self, document: DocumentMetrics):
        """Insert document metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents 
            (document_id, filename, file_size, processing_time, chunk_count, 
             embedding_time, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.document_id, document.filename, document.file_size,
            document.processing_time, document.chunk_count, document.embedding_time,
            document.success, document.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_stats(self, days: int = 30) -> pd.DataFrame:
        """Get session statistics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM sessions 
            WHERE start_time >= datetime('now', '-{} days')
            ORDER BY start_time DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_query_stats(self, days: int = 30) -> pd.DataFrame:
        """Get query statistics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM queries 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        """.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

class SystemMonitor:
    """Real-time system monitoring"""
    
    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
        """Get current system metrics"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_sessions': len(st.session_state) if hasattr(st, 'session_state') else 0
        }
    
    @staticmethod
    def log_system_metrics(db: AnalyticsDatabase):
        """Log system metrics to database"""
        metrics = SystemMonitor.get_system_metrics()
        
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics (cpu_usage, memory_usage, disk_usage, active_sessions)
            VALUES (?, ?, ?, ?)
        """, (metrics['cpu_usage'], metrics['memory_usage'], 
              metrics['disk_usage'], metrics['active_sessions']))
        
        conn.commit()
        conn.close()

class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard with comprehensive metrics"""
    
    def __init__(self):
        self.db = AnalyticsDatabase()
        self.monitor = SystemMonitor()
    
    def render_dashboard(self):
        """Render the complete analytics dashboard"""
        
        st.header("📊 Advanced Analytics Dashboard")
        
        # Time range selector
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            time_range = st.selectbox(
                "📅 Time Range",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
                index=1
            )
        
        with col2:
            refresh_rate = st.selectbox(
                "🔄 Refresh Rate",
                ["Manual", "30 seconds", "1 minute", "5 minutes"],
                index=0
            )
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Convert time range to days
        days_map = {
            "Last 24 Hours": 1,
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "All Time": 365
        }
        days = days_map[time_range]
        
        # Get data
        session_data = self.db.get_session_stats(days)
        query_data = self.db.get_query_stats(days)
        
        # Real-time system metrics
        self.render_system_metrics()
        
        # Key Performance Indicators
        self.render_kpis(session_data, query_data)
        
        # Charts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_usage_trends(query_data)
            self.render_response_time_analysis(query_data)
        
        with col2:
            self.render_model_performance(query_data)
            self.render_document_processing_stats()
        
        # Detailed tables
        self.render_detailed_analytics(session_data, query_data)
        
        # Export functionality
        self.render_export_options(session_data, query_data)
    
    def render_system_metrics(self):
        """Render real-time system metrics"""
        st.subheader("🖥️ System Performance")
        
        metrics = self.monitor.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{metrics['cpu_usage']:.1f}%",
                delta=f"{'🔴' if metrics['cpu_usage'] > 80 else '🟢'}"
            )
        
        with col2:
            st.metric(
                "Memory Usage", 
                f"{metrics['memory_usage']:.1f}%",
                delta=f"{'🔴' if metrics['memory_usage'] > 80 else '🟢'}"
            )
        
        with col3:
            st.metric(
                "Disk Usage",
                f"{metrics['disk_usage']:.1f}%",
                delta=f"{'🔴' if metrics['disk_usage'] > 90 else '🟢'}"
            )
        
        with col4:
            st.metric(
                "Active Sessions",
                metrics['active_sessions']
            )
    
    def render_kpis(self, session_data: pd.DataFrame, query_data: pd.DataFrame):
        """Render Key Performance Indicators"""
        st.subheader("📈 Key Performance Indicators")
        
        if not query_data.empty:
            # Calculate KPIs
            total_queries = len(query_data)
            avg_response_time = query_data['response_time'].mean()
            total_tokens = query_data['tokens_used'].sum()
            error_rate = (query_data['error_occurred'].sum() / total_queries * 100) if total_queries > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Queries",
                    f"{total_queries:,}",
                    delta=f"+{len(query_data[query_data['timestamp'] >= (datetime.now() - timedelta(days=1)).isoformat()])}"
                )
            
            with col2:
                st.metric(
                    "Avg Response Time",
                    f"{avg_response_time:.2f}s",
                    delta=f"{'🟢 Fast' if avg_response_time < 3 else '🔴 Slow'}"
                )
            
            with col3:
                st.metric(
                    "Total Tokens Used",
                    f"{total_tokens:,}",
                    delta=f"${total_tokens * 0.002:.2f} estimated cost"
                )
            
            with col4:
                st.metric(
                    "Error Rate",
                    f"{error_rate:.1f}%",
                    delta=f"{'🟢 Good' if error_rate < 5 else '🔴 High'}"
                )
        else:
            st.info("No query data available for the selected time range.")
    
    def render_usage_trends(self, query_data: pd.DataFrame):
        """Render usage trends chart"""
        st.subheader("📊 Usage Trends")
        
        if not query_data.empty:
            # Convert timestamp to datetime
            query_data['timestamp'] = pd.to_datetime(query_data['timestamp'])
            
            # Group by hour
            hourly_usage = query_data.groupby(query_data['timestamp'].dt.floor('H')).size().reset_index()
            hourly_usage.columns = ['hour', 'queries']
            
            fig = px.line(
                hourly_usage, 
                x='hour', 
                y='queries',
                title="Queries per Hour",
                labels={'hour': 'Time', 'queries': 'Number of Queries'}
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for usage trends.")
    
    def render_response_time_analysis(self, query_data: pd.DataFrame):
        """Render response time analysis"""
        st.subheader("⏱️ Response Time Analysis")
        
        if not query_data.empty:
            fig = go.Figure()
            
            # Response time distribution
            fig.add_trace(go.Histogram(
                x=query_data['response_time'],
                nbinsx=20,
                name="Response Time Distribution",
                opacity=0.7
            ))
            
            # Add average line
            avg_time = query_data['response_time'].mean()
            fig.add_vline(
                x=avg_time,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {avg_time:.2f}s"
            )
            
            fig.update_layout(
                title="Response Time Distribution",
                xaxis_title="Response Time (seconds)",
                yaxis_title="Frequency",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for response time analysis.")
    
    def render_model_performance(self, query_data: pd.DataFrame):
        """Render model performance comparison"""
        st.subheader("🤖 Model Performance")
        
        if not query_data.empty and 'model_used' in query_data.columns:
            model_stats = query_data.groupby('model_used').agg({
                'response_time': 'mean',
                'tokens_used': 'mean',
                'retrieval_score': 'mean'
            }).reset_index()
            
            fig = px.bar(
                model_stats,
                x='model_used',
                y='response_time',
                title="Average Response Time by Model",
                labels={'model_used': 'Model', 'response_time': 'Avg Response Time (s)'}
            )
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model performance data available.")
    
    def render_document_processing_stats(self):
        """Render document processing statistics"""
        st.subheader("📄 Document Processing")
        
        # Get document stats from database
        conn = sqlite3.connect(self.db.db_path)
        doc_data = pd.read_sql_query("SELECT * FROM documents ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        
        if not doc_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                success_rate = (doc_data['success'].sum() / len(doc_data) * 100)
                st.metric("Success Rate", f"{success_rate:.1f}%")
                
                avg_processing_time = doc_data['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
            
            with col2:
                total_chunks = doc_data['chunk_count'].sum()
                st.metric("Total Chunks", f"{total_chunks:,}")
                
                avg_file_size = doc_data['file_size'].mean() / 1024 / 1024  # MB
                st.metric("Avg File Size", f"{avg_file_size:.1f} MB")
            
            # Processing time vs file size scatter plot
            fig = px.scatter(
                doc_data,
                x='file_size',
                y='processing_time',
                color='success',
                title="Processing Time vs File Size",
                labels={'file_size': 'File Size (bytes)', 'processing_time': 'Processing Time (s)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document processing data available.")
    
    def render_detailed_analytics(self, session_data: pd.DataFrame, query_data: pd.DataFrame):
        """Render detailed analytics tables"""
        st.subheader("📋 Detailed Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Recent Queries", "Session Details", "Error Analysis"])
        
        with tab1:
            if not query_data.empty:
                # Show recent queries
                recent_queries = query_data.head(20)[['timestamp', 'query_text', 'response_time', 'model_used', 'error_occurred']]
                st.dataframe(recent_queries, use_container_width=True)
            else:
                st.info("No recent queries to display.")
        
        with tab2:
            if not session_data.empty:
                st.dataframe(session_data, use_container_width=True)
            else:
                st.info("No session data to display.")
        
        with tab3:
            if not query_data.empty:
                error_queries = query_data[query_data['error_occurred'] == True]
                if not error_queries.empty:
                    st.dataframe(error_queries[['timestamp', 'query_text', 'model_used']], use_container_width=True)
                else:
                    st.success("No errors in the selected time range! 🎉")
            else:
                st.info("No error data to analyze.")
    
    def render_export_options(self, session_data: pd.DataFrame, query_data: pd.DataFrame):
        """Render data export options"""
        st.subheader("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Query Data"):
                if not query_data.empty:
                    csv = query_data.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "query_analytics.csv",
                        "text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Session Data"):
                if not session_data.empty:
                    csv = session_data.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        "session_analytics.csv", 
                        "text/csv"
                    )
        
        with col3:
            if st.button("📋 Export Full Report"):
                # Create comprehensive report
                report = self.generate_analytics_report(session_data, query_data)
                st.download_button(
                    "Download Report",
                    report,
                    "analytics_report.json",
                    "application/json"
                )
    
    def generate_analytics_report(self, session_data: pd.DataFrame, query_data: pd.DataFrame) -> str:
        """Generate comprehensive analytics report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_sessions": len(session_data),
                "total_queries": len(query_data),
                "avg_response_time": query_data['response_time'].mean() if not query_data.empty else 0,
                "error_rate": (query_data['error_occurred'].sum() / len(query_data) * 100) if not query_data.empty else 0
            },
            "session_data": session_data.to_dict('records') if not session_data.empty else [],
            "query_data": query_data.to_dict('records') if not query_data.empty else [],
            "system_metrics": self.monitor.get_system_metrics()
        }
        
        return json.dumps(report, indent=2, default=str)

# Streamlit integration functions
def init_analytics():
    """Initialize analytics system"""
    if 'analytics_db' not in st.session_state:
        st.session_state.analytics_db = AnalyticsDatabase()
    
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()

def track_query(query_text: str, response_time: float, model_used: str, 
                tokens_used: int = 0, retrieval_score: float = 0.0, 
                error_occurred: bool = False):
    """Track individual query metrics"""
    if 'analytics_db' not in st.session_state:
        init_analytics()
    
    query_metrics = QueryMetrics(
        query_id=f"query_{int(time.time() * 1000)}",
        session_id=st.session_state.get('session_id', 'default'),
        timestamp=datetime.now(),
        query_text=query_text,
        response_time=response_time,
        tokens_used=tokens_used,
        model_used=model_used,
        retrieval_score=retrieval_score,
        user_rating=None,
        error_occurred=error_occurred
    )
    
    st.session_state.analytics_db.insert_query(query_metrics)

def track_document_processing(filename: str, file_size: int, processing_time: float,
                            chunk_count: int, success: bool, error_message: str = None):
    """Track document processing metrics"""
    if 'analytics_db' not in st.session_state:
        init_analytics()
    
    doc_metrics = DocumentMetrics(
        document_id=f"doc_{int(time.time() * 1000)}",
        filename=filename,
        file_size=file_size,
        processing_time=processing_time,
        chunk_count=chunk_count,
        embedding_time=0.0,  # Would be calculated separately
        success=success,
        error_message=error_message
    )
    
    st.session_state.analytics_db.insert_document(doc_metrics)

# Export main classes and functions
__all__ = [
    'AdvancedAnalyticsDashboard',
    'AnalyticsDatabase',
    'SystemMonitor',
    'SessionMetrics',
    'QueryMetrics', 
    'DocumentMetrics',
    'init_analytics',
    'track_query',
    'track_document_processing'
]