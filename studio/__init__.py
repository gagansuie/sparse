"""
TenPak Studio - API and Job Management

Provides REST API for compression jobs and artifact management.
Designed for integration with HF Hub and Inference Endpoints.

Usage:
    # Start the API server
    uvicorn tenpak.studio.api:app --host 0.0.0.0 --port 8000
    
    # Or programmatically
    from tenpak.studio import create_job, get_job_status
"""

from .api import app
from .jobs import CompressionJob, JobStatus, create_job, get_job_status

__all__ = [
    "app",
    "CompressionJob",
    "JobStatus",
    "create_job",
    "get_job_status",
]
