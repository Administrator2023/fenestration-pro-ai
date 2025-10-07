"""
Performance Optimization Module
Advanced caching, async processing, and performance optimizations
"""

import asyncio
import aiofiles
import aioredis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import threading
from pathlib import Path
import psutil
import gc
import sys
from dataclasses import dataclass
from contextlib import asynccontextmanager

import streamlit as st
import pandas as pd
import numpy as np
from memory_profiler import profile
import cProfile
import pstats
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation_name: str
    start_time: float
    end_time: float
    memory_before: float
    memory_after: float
    cpu_usage: float
    cache_hit: bool = False
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def memory_delta(self) -> float:
        return self.memory_after - self.memory_before

class AdvancedCache:
    """Advanced caching system with multiple backends"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 3600,
                 max_memory_items: int = 1000):
        self.default_ttl = default_ttl
        self.max_memory_items = max_memory_items
        
        # Memory cache
        self.memory_cache = {}
        self.memory_access_times = {}
        self.memory_lock = threading.RLock()
        
        # Redis cache (optional)
        self.redis_client = None
        self.redis_available = False
        
        # Initialize Redis if available
        asyncio.create_task(self._init_redis(redis_url))
    
    async def _init_redis(self, redis_url: str):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache only: {e}")
            self.redis_available = False
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        # Create a hashable representation of arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        # Try Redis first if available
        if self.redis_available and self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Fallback to memory cache
        with self.memory_lock:
            if key in self.memory_cache:
                self.memory_access_times[key] = time.time()
                return self.memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        ttl = ttl or self.default_ttl
        
        # Try Redis first if available
        if self.redis_available and self.redis_client:
            try:
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Fallback to memory cache
        with self.memory_lock:
            # Implement LRU eviction if memory cache is full
            if len(self.memory_cache) >= self.max_memory_items:
                self._evict_lru()
            
            self.memory_cache[key] = value
            self.memory_access_times[key] = time.time()
        
        return True
    
    def _evict_lru(self):
        """Evict least recently used items from memory cache"""
        if not self.memory_access_times:
            return
        
        # Find least recently used key
        lru_key = min(self.memory_access_times, key=self.memory_access_times.get)
        
        # Remove from both caches
        self.memory_cache.pop(lru_key, None)
        self.memory_access_times.pop(lru_key, None)
    
    async def clear(self):
        """Clear all caches"""
        if self.redis_available and self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        with self.memory_lock:
            self.memory_cache.clear()
            self.memory_access_times.clear()

# Global cache instance
cache = AdvancedCache()

def async_cached(ttl: int = 3600, key_prefix: str = ""):
    """Async caching decorator"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
            cache_key = cache._generate_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for {func_name}, executing function")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def cached(ttl: int = 3600, key_prefix: str = ""):
    """Synchronous caching decorator"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
            cache_key = cache._generate_key(func_name, args, kwargs)
            
            # Try to get from cache (blocking)
            try:
                loop = asyncio.get_event_loop()
                cached_result = loop.run_until_complete(cache.get(cache_key))
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func_name}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
            
            # Execute function
            logger.debug(f"Cache miss for {func_name}, executing function")
            result = func(*args, **kwargs)
            
            # Store in cache (non-blocking)
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(cache.set(cache_key, result, ttl))
            except Exception as e:
                logger.warning(f"Cache set error: {e}")
            
            return result
        
        return wrapper
    return decorator

class AsyncProcessor:
    """Async processing utilities"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, psutil.cpu_count() or 1))
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, functools.partial(func, **kwargs), *args)
    
    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive function in process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_executor, functools.partial(func, **kwargs), *args)
    
    async def batch_process(self, 
                          func: Callable, 
                          items: List[Any], 
                          batch_size: int = 10,
                          use_processes: bool = False) -> List[Any]:
        """Process items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if use_processes:
                batch_tasks = [self.run_in_process(func, item) for item in batch]
            else:
                batch_tasks = [self.run_in_thread(func, item) for item in batch]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def cleanup(self):
        """Cleanup executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

# Global async processor
async_processor = AsyncProcessor()

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.metrics = []
        self.profiler = None
    
    @asynccontextmanager
    async def monitor(self, operation_name: str):
        """Context manager for monitoring performance"""
        # Get initial metrics
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        start_time = time.time()
        
        try:
            yield
        finally:
            # Get final metrics
            end_time = time.time()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                memory_before=memory_before,
                memory_after=memory_after,
                cpu_usage=(cpu_before + cpu_after) / 2
            )
            
            self.metrics.append(metrics)
            
            logger.info(f"Performance: {operation_name} took {metrics.duration:.2f}s, "
                       f"memory delta: {metrics.memory_delta:.2f}MB, "
                       f"CPU: {metrics.cpu_usage:.1f}%")
    
    def start_profiling(self):
        """Start CPU profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop_profiling(self) -> str:
        """Stop profiling and return results"""
        if not self.profiler:
            return "No profiling session active"
        
        self.profiler.disable()
        
        # Get profiling results
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return s.getvalue()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        durations = [m.duration for m in self.metrics]
        memory_deltas = [m.memory_delta for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "operations": [
                {
                    "name": m.operation_name,
                    "duration": m.duration,
                    "memory_delta": m.memory_delta,
                    "cpu_usage": m.cpu_usage
                }
                for m in self.metrics[-10:]  # Last 10 operations
            ]
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage"""
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame memory optimized: {reduction:.1f}% reduction")
        
        return df
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup"""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }

class StreamlitOptimizer:
    """Streamlit-specific optimizations"""
    
    @staticmethod
    def optimize_session_state():
        """Optimize Streamlit session state"""
        if not hasattr(st, 'session_state'):
            return
        
        # Clean up old data
        current_time = time.time()
        keys_to_remove = []
        
        for key, value in st.session_state.items():
            if hasattr(value, 'timestamp'):
                # Remove data older than 1 hour
                if current_time - value.timestamp > 3600:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old session state items")
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=100)
    def cached_dataframe_operation(df: pd.DataFrame, operation: str) -> pd.DataFrame:
        """Cached DataFrame operations"""
        if operation == "optimize":
            return MemoryOptimizer.optimize_dataframe(df.copy())
        elif operation == "describe":
            return df.describe()
        else:
            return df
    
    @staticmethod
    @st.cache_resource(ttl=3600)
    def cached_model_loading(model_name: str):
        """Cache expensive model loading operations"""
        # This would load and cache ML models
        logger.info(f"Loading model: {model_name}")
        return f"cached_model_{model_name}"

class BatchProcessor:
    """Batch processing for large datasets"""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    async def process_dataframe_batches(self, 
                                      df: pd.DataFrame, 
                                      process_func: Callable,
                                      **kwargs) -> pd.DataFrame:
        """Process DataFrame in batches"""
        results = []
        
        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i:i + self.batch_size]
            
            async with perf_monitor.monitor(f"batch_process_{i}"):
                if asyncio.iscoroutinefunction(process_func):
                    result = await process_func(batch, **kwargs)
                else:
                    result = await async_processor.run_in_thread(process_func, batch, **kwargs)
                
                results.append(result)
        
        return pd.concat(results, ignore_index=True)
    
    async def process_files_batches(self, 
                                   file_paths: List[str], 
                                   process_func: Callable,
                                   **kwargs) -> List[Any]:
        """Process files in batches"""
        results = []
        
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]
            
            async with perf_monitor.monitor(f"file_batch_process_{i}"):
                batch_tasks = []
                for file_path in batch:
                    if asyncio.iscoroutinefunction(process_func):
                        task = process_func(file_path, **kwargs)
                    else:
                        task = async_processor.run_in_thread(process_func, file_path, **kwargs)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
        
        return results

# Performance optimization decorators
def optimize_memory(func: Callable):
    """Decorator to optimize memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Cleanup before execution
        MemoryOptimizer.cleanup_memory()
        
        result = func(*args, **kwargs)
        
        # Cleanup after execution
        MemoryOptimizer.cleanup_memory()
        
        return result
    
    return wrapper

def monitor_performance(operation_name: str = None):
    """Decorator to monitor performance"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            async with perf_monitor.monitor(name):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            # For sync functions, we'll use a simpler monitoring approach
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Performance: {name} took {duration:.2f}s")
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Streamlit integration functions
def init_performance_optimization():
    """Initialize performance optimizations for Streamlit"""
    if 'perf_optimizer_initialized' not in st.session_state:
        # Optimize session state
        StreamlitOptimizer.optimize_session_state()
        
        # Set up periodic cleanup
        if 'last_cleanup' not in st.session_state:
            st.session_state.last_cleanup = time.time()
        
        # Cleanup every 5 minutes
        if time.time() - st.session_state.last_cleanup > 300:
            MemoryOptimizer.cleanup_memory()
            StreamlitOptimizer.optimize_session_state()
            st.session_state.last_cleanup = time.time()
        
        st.session_state.perf_optimizer_initialized = True

def render_performance_dashboard():
    """Render performance monitoring dashboard"""
    st.subheader("⚡ Performance Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    memory_usage = MemoryOptimizer.get_memory_usage()
    
    with col1:
        st.metric("Memory Usage", f"{memory_usage['rss_mb']:.1f} MB", 
                 f"{memory_usage['percent']:.1f}%")
    
    with col2:
        cpu_usage = psutil.cpu_percent()
        st.metric("CPU Usage", f"{cpu_usage:.1f}%")
    
    with col3:
        disk_usage = psutil.disk_usage('/').percent
        st.metric("Disk Usage", f"{disk_usage:.1f}%")
    
    with col4:
        cache_size = len(cache.memory_cache)
        st.metric("Cache Items", cache_size)
    
    # Performance metrics
    metrics_summary = perf_monitor.get_metrics_summary()
    
    if metrics_summary.get("total_operations", 0) > 0:
        st.subheader("📊 Operation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Operations", metrics_summary["total_operations"])
            st.metric("Avg Duration", f"{metrics_summary['avg_duration']:.2f}s")
        
        with col2:
            st.metric("Max Duration", f"{metrics_summary['max_duration']:.2f}s")
            st.metric("Avg Memory Delta", f"{metrics_summary['avg_memory_delta']:.2f}MB")
        
        # Recent operations table
        if metrics_summary.get("operations"):
            st.subheader("Recent Operations")
            df = pd.DataFrame(metrics_summary["operations"])
            st.dataframe(df, use_container_width=True)
    
    # Cache management
    st.subheader("🗄️ Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Cache"):
            asyncio.run(cache.clear())
            st.success("Cache cleared!")
    
    with col2:
        if st.button("Force Garbage Collection"):
            MemoryOptimizer.cleanup_memory()
            st.success("Memory cleaned up!")

# Export main classes and functions
__all__ = [
    'AdvancedCache',
    'AsyncProcessor', 
    'PerformanceMonitor',
    'MemoryOptimizer',
    'StreamlitOptimizer',
    'BatchProcessor',
    'async_cached',
    'cached',
    'optimize_memory',
    'monitor_performance',
    'init_performance_optimization',
    'render_performance_dashboard',
    'cache',
    'async_processor',
    'perf_monitor'
]