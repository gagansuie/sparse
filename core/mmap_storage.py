"""
Memory-mapped delta storage for fast I/O.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch


class MmapDeltaStorage:
    """Memory-mapped storage for delta tensors."""
    
    def __init__(self, base_path: Path):
        """
        Args:
            base_path: Base directory for delta storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
    
    def save_delta(
        self,
        name: str,
        delta: torch.Tensor,
        quantized: Optional[np.ndarray] = None,
        scale: Optional[float] = None,
    ) -> str:
        """
        Save delta using memory-mapped file.
        
        Args:
            name: Layer name
            delta: Delta tensor
            quantized: Optional quantized data
            scale: Optional quantization scale
        
        Returns:
            Path to saved file
        """
        # Sanitize name for filename
        safe_name = name.replace("/", "_").replace(".", "_")
        delta_file = self.base_path / f"{safe_name}.bin"
        
        if quantized is not None:
            # Save quantized INT8 data
            data = quantized.astype(np.int8)
            dtype = np.int8
        else:
            # Save FP16 data
            data = delta.cpu().numpy().astype(np.float16)
            dtype = np.float16
        
        # Create memory-mapped file
        mmap = np.memmap(
            delta_file,
            dtype=dtype,
            mode='w+',
            shape=data.shape
        )
        mmap[:] = data
        mmap.flush()
        
        # Save metadata
        self.metadata[name] = {
            "file": str(delta_file),
            "shape": list(data.shape),
            "dtype": dtype.__name__ if hasattr(dtype, '__name__') else str(dtype),
            "scale": scale,
        }
        
        return str(delta_file)
    
    def load_delta(
        self,
        name: str,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Load delta from memory-mapped file.
        
        Args:
            name: Layer name
            device: Device to load to
        
        Returns:
            Tuple of (delta tensor, scale if quantized)
        """
        if name not in self.metadata:
            raise KeyError(f"Delta {name} not found")
        
        meta = self.metadata[name]
        
        # Memory-map the file
        mmap = np.memmap(
            meta["file"],
            dtype=np.dtype(meta["dtype"]),
            mode='r',
            shape=tuple(meta["shape"])
        )
        
        # Convert to tensor (zero-copy if possible)
        tensor = torch.from_numpy(np.array(mmap)).to(device)
        
        return tensor, meta.get("scale")
    
    def save_metadata(self, metadata_file: Path):
        """Save metadata to JSON file."""
        import json
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_metadata(self, metadata_file: Path):
        """Load metadata from JSON file."""
        import json
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
    
    def get_total_size_mb(self) -> float:
        """Get total size of all delta files in MB."""
        total_bytes = sum(
            Path(meta["file"]).stat().st_size
            for meta in self.metadata.values()
            if Path(meta["file"]).exists()
        )
        return total_bytes / (1024 * 1024)
