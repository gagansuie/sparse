"""
Sparse CLI - Command-line interface for delta compression and smart routing

Usage:
    sparse delta compress <base> <finetune> --output <path>
    sparse delta-dataset compress <base> <derivative> --output <path>
    sparse route <model> <prompt>
    sparse optimize <model_id> --max-ppl-delta 2.0
    sparse eval <model_id> [--samples N]
    
Examples:
    # Model delta compression
    sparse delta compress meta-llama/Llama-2-7b-hf my-org/llama-chat --output ./delta
    
    # Dataset delta compression
    sparse delta-dataset compress squad squad_v2 --output ./dataset_delta
    
    # Smart routing
    sparse route meta-llama/Llama-2-70b-hf "What is the capital of France?"
    
    # Cost optimizer
    sparse optimize mistralai/Mistral-7B-v0.1 --max-ppl-delta 2.0
    
    # Evaluate perplexity
    sparse eval TinyLlama/TinyLlama-1.1B-Chat-v1.0 --samples 100
"""

import argparse
import sys
from .main import main

__all__ = ["main"]
