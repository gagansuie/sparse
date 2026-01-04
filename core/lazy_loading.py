"""
Lazy loading with safetensors for memory-efficient processing.
"""

import torch
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional
from safetensors.torch import load_file, save_file
from transformers import AutoConfig


class LazyModelLoader:
    """Load model weights layer-by-layer without loading entire model."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to load tensors to
        """
        self.model_path = model_path
        self.device = device
        self.config = AutoConfig.from_pretrained(model_path)
        self._weights_file = None
        self._find_weights_file()
    
    def _find_weights_file(self):
        """Find safetensors or pytorch weights file."""
        model_dir = Path(self.model_path)
        
        if not model_dir.exists():
            # Try to download from HF
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            # Convert to safetensors
            temp_dir = Path("/tmp/sparse_lazy_cache") / self.model_path.replace("/", "_")
            temp_dir.mkdir(parents=True, exist_ok=True)
            self._weights_file = temp_dir / "model.safetensors"
            save_file(dict(model.named_parameters()), self._weights_file)
            del model
            torch.cuda.empty_cache()
            return
        
        # Look for safetensors file
        safetensors_files = list(model_dir.glob("*.safetensors"))
        if safetensors_files:
            self._weights_file = safetensors_files[0]
        else:
            # Fall back to pytorch_model.bin
            bin_files = list(model_dir.glob("pytorch_model*.bin"))
            if bin_files:
                # Convert to safetensors
                weights = torch.load(bin_files[0], map_location="cpu")
                self._weights_file = model_dir / "model.safetensors"
                save_file(weights, self._weights_file)
            else:
                raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    def iter_layers(
        self,
        layer_names: Optional[list] = None
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over model layers without loading all into memory.
        
        Args:
            layer_names: Optional list of specific layers to load
        
        Yields:
            Tuples of (layer_name, tensor)
        """
        # Load weights metadata
        all_weights = load_file(str(self._weights_file), device=self.device)
        
        if layer_names is None:
            layer_names = list(all_weights.keys())
        
        for name in layer_names:
            if name in all_weights:
                tensor = all_weights[name]
                yield name, tensor
                # Free memory after yielding
                del tensor
                if self.device == "cuda":
                    torch.cuda.empty_cache()
    
    def load_layer(self, layer_name: str) -> torch.Tensor:
        """Load a single layer."""
        weights = load_file(str(self._weights_file), device=self.device)
        if layer_name not in weights:
            raise KeyError(f"Layer {layer_name} not found")
        return weights[layer_name]
    
    def get_layer_names(self) -> list:
        """Get list of all layer names without loading weights."""
        from safetensors import safe_open
        with safe_open(str(self._weights_file), framework="pt", device="cpu") as f:
            return list(f.keys())


def compute_deltas_streaming(
    base_loader: LazyModelLoader,
    finetune_loader: LazyModelLoader,
) -> Iterator[Tuple[str, torch.Tensor, Dict]]:
    """
    Compute deltas in streaming fashion, processing one layer at a time.
    
    Args:
        base_loader: Lazy loader for base model
        finetune_loader: Lazy loader for fine-tuned model
    
    Yields:
        Tuples of (name, delta, stats)
    """
    base_names = set(base_loader.get_layer_names())
    finetune_names = set(finetune_loader.get_layer_names())
    common_names = base_names & finetune_names
    
    for name in common_names:
        # Load only this layer from both models
        base_weight = base_loader.load_layer(name)
        finetune_weight = finetune_loader.load_layer(name)
        
        if base_weight.shape != finetune_weight.shape:
            continue
        
        # Compute delta
        delta = finetune_weight - base_weight
        
        # Compute stats
        stats = {
            "max_abs": delta.abs().max().item(),
            "l2_norm": delta.norm().item(),
            "sparsity": (delta.abs() < 1e-6).float().mean().item(),
            "numel": delta.numel(),
        }
        
        yield name, delta, stats
        
        # Free memory
        del base_weight, finetune_weight, delta
        torch.cuda.empty_cache()
