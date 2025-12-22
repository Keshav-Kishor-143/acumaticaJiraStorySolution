#!/usr/bin/env python3
"""
Logger utilities for Story Solutions Service
"""

import logging
import sys
import datetime
from typing import Optional, Dict, Any

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger instance with consistent formatting"""
    logger = logging.getLogger(name)
    
    # Set default level to DEBUG
    if level is None:
        level = logging.DEBUG
    
    logger.setLevel(level)
    
    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        class SafeLogRecord(logging.LogRecord):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.extra_data = {}

        class SafeFormatter(logging.Formatter):
            def format(self, record):
                # Safely handle extra parameters
                if hasattr(record, 'extra') and isinstance(record.extra, dict):
                    # Store extra data separately to avoid conflicts
                    record.extra_data = record.extra
                    # Format extra data as a string
                    extra_str = ' '.join(f"{k}={v}" for k, v in record.extra.items())
                    record.extra_str = f"[{extra_str}]" if extra_str else ""
                else:
                    record.extra_str = ""
                return super().format(record)

        formatter = SafeFormatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s%(extra_str)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics for operations"""
    
    def __init__(self):
        self.metrics = {
            "story_processing": [],
            "question_extraction": [],
            "rag_search": [],
            "solution_generation": [],
            "total_processing_time": []
        }
    
    def record_metric(self, operation: str, duration: float, details: Optional[Dict] = None):
        """Record a performance metric"""
        metric_entry = {
            "timestamp": datetime.datetime.now(),
            "duration": duration,
            "details": details or {}
        }
        
        if operation in self.metrics:
            self.metrics[operation].append(metric_entry)
            
            # Keep only recent metrics (last 100)
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
    
    def get_average_duration(self, operation: str, last_n: int = 10) -> float:
        """Get average duration for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        recent_metrics = self.metrics[operation][-last_n:]
        durations = [m["duration"] for m in recent_metrics]
        return sum(durations) / len(durations) if durations else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {}
        for operation, metrics_list in self.metrics.items():
            if metrics_list:
                durations = [m["duration"] for m in metrics_list]
                summary[operation] = {
                    "count": len(durations),
                    "average": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "recent_average": self.get_average_duration(operation, 5)
                }
            else:
                summary[operation] = {
                    "count": 0,
                    "average": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "recent_average": 0.0
                }
        return summary

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Context manager for timing operations
class TimedOperation:
    """Context manager for timing operations with automatic logging"""
    
    def __init__(self, operation_name: str, logger: logging.Logger, **details):
        self.operation_name = operation_name
        self.logger = logger
        self.details = details
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.datetime.now()
        self.logger.info(f"Starting {self.operation_name}...", extra=self.details)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.now()
        self.duration = (end_time - self.start_time).total_seconds()
        
        # Record performance metric
        performance_monitor.record_metric(self.operation_name, self.duration, self.details)
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation_name}",
                extra={"duration": f"{self.duration:.2f}s", **self.details}
            )
        else:
            self.logger.error(
                f"Failed {self.operation_name}: {exc_val}",
                extra={"duration": f"{self.duration:.2f}s", **self.details}
            )

