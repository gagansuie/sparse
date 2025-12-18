"""
TenPak CLI - Command-line interface for model compression

Usage:
    tenpak pack <model_id> [--target quality|balanced|size] [--output path]
    tenpak eval <model_id> [--samples N]
    tenpak info <artifact_path>
    
Examples:
    # Compress a model with balanced settings
    tenpak pack mistralai/Mistral-7B-v0.1 --target balanced
    
    # Evaluate baseline perplexity
    tenpak eval TinyLlama/TinyLlama-1.1B-Chat-v1.0 --samples 100
    
    # Get info about a compressed artifact
    tenpak info /path/to/artifact
"""

import argparse
import sys
from .main import main

__all__ = ["main"]
