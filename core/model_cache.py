"""
Base Model Cache for faster compression of multiple fine-tunes.
"""

import os
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM


class BaseModelCache:
    """LRU cache for base models to avoid repeated loading."""
    
    def __init__(self, max_size_gb: float = 50.0, cache_dir: Optional[str] = None):
        """
        Args:
            max_size_gb: Maximum cache size in gigabytes
            cache_dir: Optional directory for disk cache
        """
        self.max_size_gb = max_size_gb
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, Tuple[any, float, float]] = {}  # model_id -> (model, size_gb, last_used)
        self._total_size_gb = 0.0
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_size_gb(self, model) -> float:
        """Estimate model size in GB."""
        total_params = sum(p.numel() for p in model.parameters())
        # FP16 = 2 bytes per param
        return (total_params * 2) / (1024 ** 3)
    
    def _evict_lru(self, required_size_gb: float):
        """Evict least recently used models to make space."""
        while self._total_size_gb + required_size_gb > self.max_size_gb and self._cache:
            # Find LRU model
            lru_model_id = min(self._cache.items(), key=lambda x: x[1][2])[0]
            _, size_gb, _ = self._cache.pop(lru_model_id)
            self._total_size_gb -= size_gb
            print(f"[Cache] Evicted {lru_model_id} ({size_gb:.2f} GB)")
    
    def get_or_load(
        self,
        model_id: str,
        torch_dtype=torch.float16,
        device_map: str = "cpu",
        low_cpu_mem_usage: bool = True,
    ):
        """Get model from cache or load if not present."""
        # Check cache
        if model_id in self._cache:
            model, size_gb, _ = self._cache[model_id]
            self._cache[model_id] = (model, size_gb, time.time())
            print(f"\033[92m[Cache] Hit: {model_id}\033[0m")
            return model
        
        # Cache miss - load model
        print(f"\033[93m[Cache] Miss: {model_id} - loading...\033[0m")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        
        size_gb = self.get_model_size_gb(model)
        
        # Evict if needed
        self._evict_lru(size_gb)
        
        # Add to cache
        self._cache[model_id] = (model, size_gb, time.time())
        self._total_size_gb += size_gb
        
        print(f"\033[94m[Cache] Loaded {model_id} ({size_gb:.2f} GB) - Total: {self._total_size_gb:.2f}/{self.max_size_gb} GB\033[0m")
        
        return model
    
    def clear(self):
        """Clear all cached models."""
        print("\033[91m[Cache] Clearing cache...\033[0m")
        self._cache.clear()
        self._total_size_gb = 0.0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_models": list(self._cache.keys()),
            "total_size_gb": self._total_size_gb,
            "max_size_gb": self.max_size_gb,
            "utilization": self._total_size_gb / self.max_size_gb if self.max_size_gb > 0 else 0,
        }


# Global cache instance
_global_cache: Optional[BaseModelCache] = None


def get_cache(max_size_gb: float = 50.0) -> BaseModelCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = BaseModelCache(max_size_gb=max_size_gb)
    return _global_cache
