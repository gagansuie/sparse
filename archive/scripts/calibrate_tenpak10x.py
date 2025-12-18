#!/usr/bin/env python3
"""
TenPak-10X Calibration Script

Collects Fisher information and learns shared codebooks for 10x compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
import os


@dataclass
class LayerAllocation:
    """Bit allocation for a single layer."""
    name: str
    method: str  # 'lowrank_int4', 'vq_int2', 'vq_only'
    importance: float
    rank: int = 32
    group_size: int = 16
    codebook_id: str = 'medium'
    bits_per_weight: float = 3.5


class TenPak10XCalibrator:
    """Calibration and compression for TenPak-10X."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.fisher_scores = {}
        self.activation_stats = {}
        self.allocations = {}
        self.codebooks = {}
        
    def collect_fisher_info(self, texts: List[str], num_samples: int = 128) -> Dict[str, float]:
        """Collect Fisher information scores for each layer."""
        print(f"[CALIBRATE] Collecting Fisher information from {num_samples} samples...")
        
        self.model.train()  # Need gradients
        fisher_accum = {}
        
        for i, text in enumerate(tqdm(texts[:num_samples], desc="Fisher")):
            # Tokenize
            tokens = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            if tokens.input_ids.shape[1] < 2:
                continue
            
            # Forward + backward
            try:
                outputs = self.model(**tokens, labels=tokens.input_ids)
                loss = outputs.loss
                loss.backward()
                
                # Accumulate gradient squared (Fisher approximation)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and 'weight' in name:
                        grad_sq = param.grad.pow(2).mean().item()
                        fisher_accum[name] = fisher_accum.get(name, 0) + grad_sq
                
                self.model.zero_grad()
                
            except Exception as e:
                print(f"[CALIBRATE] Error at sample {i}: {e}")
                continue
            
            # Memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
        
        # Normalize scores
        if fisher_accum:
            max_score = max(fisher_accum.values())
            self.fisher_scores = {k: v / max_score for k, v in fisher_accum.items()}
        
        self.model.eval()
        print(f"[CALIBRATE] Collected Fisher scores for {len(self.fisher_scores)} layers")
        return self.fisher_scores
    
    def allocate_bits(self, target_avg_bits: float = 3.2) -> Dict[str, LayerAllocation]:
        """Allocate bits per layer based on Fisher importance."""
        print(f"[CALIBRATE] Allocating bits (target: {target_avg_bits} bits/weight)...")
        
        allocations = {}
        
        # Get linear layers only
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'embed' not in name.lower():
                fisher = self.fisher_scores.get(f"{name}.weight", 0.1)
                linear_layers.append((name, module, fisher))
        
        # Sort by importance
        linear_layers.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate based on importance tiers
        n_layers = len(linear_layers)
        critical_cutoff = int(n_layers * 0.15)  # Top 15% = critical
        medium_cutoff = int(n_layers * 0.50)    # Next 35% = medium
        
        for i, (name, module, importance) in enumerate(linear_layers):
            if i < critical_cutoff:
                # Critical: low-rank + INT4 residual (~6 bits)
                alloc = LayerAllocation(
                    name=name,
                    method='lowrank_int4',
                    importance=importance,
                    rank=64,
                    group_size=8,
                    codebook_id='critical',
                    bits_per_weight=6.0
                )
            elif i < medium_cutoff:
                # Medium: VQ + INT2 residual (~3.5 bits)
                alloc = LayerAllocation(
                    name=name,
                    method='vq_int2',
                    importance=importance,
                    rank=32,
                    group_size=32,
                    codebook_id='medium',
                    bits_per_weight=3.5
                )
            else:
                # Robust: VQ only (~2 bits)
                alloc = LayerAllocation(
                    name=name,
                    method='vq_only',
                    importance=importance,
                    codebook_id='aggressive',
                    bits_per_weight=2.0
                )
            
            allocations[name] = alloc
        
        self.allocations = allocations
        
        # Calculate expected compression
        total_params = sum(m.weight.numel() for _, m, _ in linear_layers)
        total_bits = sum(a.bits_per_weight * self._get_layer_params(a.name) 
                        for a in allocations.values())
        avg_bits = total_bits / total_params
        compression = 32.0 / avg_bits
        
        print(f"[CALIBRATE] Allocated {len(allocations)} layers")
        print(f"[CALIBRATE] Expected avg bits: {avg_bits:.2f}, compression: {compression:.1f}x")
        
        return allocations
    
    def _get_layer_params(self, name: str) -> int:
        """Get number of parameters in a layer."""
        for n, m in self.model.named_modules():
            if n == name and hasattr(m, 'weight'):
                return m.weight.numel()
        return 0
    
    def learn_shared_codebooks(
        self, 
        texts: List[str],
        num_epochs: int = 50,
        lr: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Learn shared codebooks via gradient descent on reconstruction loss."""
        print(f"[CALIBRATE] Learning shared codebooks ({num_epochs} epochs)...")
        
        # Initialize codebooks
        codebooks = nn.ParameterDict({
            'critical': nn.Parameter(torch.randn(512, 4, device=self.device) * 0.02),
            'medium': nn.Parameter(torch.randn(256, 8, device=self.device) * 0.02),
            'aggressive': nn.Parameter(torch.randn(128, 16, device=self.device) * 0.02),
        })
        
        # Collect weight statistics for initialization
        all_weights = []
        for name, alloc in self.allocations.items():
            for n, m in self.model.named_modules():
                if n == name and hasattr(m, 'weight'):
                    all_weights.append(m.weight.data.float().cpu())
                    break
        
        if all_weights:
            # Initialize codebooks from weight statistics
            all_flat = torch.cat([w.flatten() for w in all_weights])
            std = all_flat.std().item()
            
            for cb in codebooks.values():
                cb.data = cb.data * std
        
        optimizer = torch.optim.Adam(codebooks.parameters(), lr=lr)
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for name, alloc in self.allocations.items():
                if alloc.method == 'lowrank_int4':
                    continue  # Skip low-rank layers for codebook learning
                
                # Get weight
                weight = None
                for n, m in self.model.named_modules():
                    if n == name and hasattr(m, 'weight'):
                        weight = m.weight.data.float()
                        break
                
                if weight is None:
                    continue
                
                # Get codebook
                cb = codebooks[alloc.codebook_id]
                vec_dim = cb.shape[1]
                
                # Reshape weight to vectors
                flat = weight.flatten()
                pad_len = (vec_dim - len(flat) % vec_dim) % vec_dim
                if pad_len > 0:
                    flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])
                vectors = flat.view(-1, vec_dim)
                
                # Soft quantization (differentiable)
                # Distance to each codebook entry
                dists = torch.cdist(vectors, cb)  # [num_vecs, num_entries]
                
                # Soft assignment (temperature annealing)
                temp = max(0.1, 1.0 - epoch / num_epochs)
                soft_assign = F.softmax(-dists / temp, dim=1)
                
                # Reconstructed vectors
                reconstructed = soft_assign @ cb
                
                # Reconstruction loss
                loss = F.mse_loss(reconstructed, vectors)
                total_loss += loss.item()
                num_batches += 1
                
                # Backward
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = total_loss / max(num_batches, 1)
            if epoch % 10 == 0:
                print(f"[CALIBRATE] Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        # Convert to regular tensors
        self.codebooks = {k: v.data.clone() for k, v in codebooks.items()}
        print(f"[CALIBRATE] Codebook learning complete. Final loss: {best_loss:.6f}")
        
        return self.codebooks
    
    def save_calibration(self, path: str):
        """Save calibration data to file."""
        data = {
            'fisher_scores': self.fisher_scores,
            'allocations': {k: vars(v) for k, v in self.allocations.items()},
            'codebooks': {k: v.cpu().tolist() for k, v in self.codebooks.items()},
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"[CALIBRATE] Saved calibration to {path}")
    
    def load_calibration(self, path: str):
        """Load calibration data from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.fisher_scores = data['fisher_scores']
        self.allocations = {
            k: LayerAllocation(**v) for k, v in data['allocations'].items()
        }
        self.codebooks = {
            k: torch.tensor(v, device=self.device) for k, v in data['codebooks'].items()
        }
        
        print(f"[CALIBRATE] Loaded calibration from {path}")


class TenPak10XCompressor:
    """Compress and decompress using TenPak-10X."""
    
    def __init__(self, calibrator: TenPak10XCalibrator):
        self.calibrator = calibrator
        self.device = calibrator.device
    
    def compress_layer(self, weight: torch.Tensor, alloc: LayerAllocation) -> Dict:
        """Compress a single layer."""
        weight = weight.float()
        
        if alloc.method == 'lowrank_int4':
            return self._compress_lowrank_int4(weight, alloc)
        elif alloc.method == 'vq_int2':
            return self._compress_vq_int2(weight, alloc)
        else:  # vq_only
            return self._compress_vq_only(weight, alloc)
    
    def _compress_lowrank_int4(self, weight: torch.Tensor, alloc: LayerAllocation) -> Dict:
        """Low-rank + INT4 residual compression."""
        rank = alloc.rank
        
        # SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        # Truncate to rank
        L = U[:, :rank] @ torch.diag(S[:rank].sqrt())
        R = torch.diag(S[:rank].sqrt()) @ Vh[:rank, :]
        
        # Compute residual
        approx = L @ R
        residual = weight - approx
        
        # INT4 quantize residual
        residual_q, scale, offset = self._int4_quantize(residual, alloc.group_size)
        
        return {
            'method': 'lowrank_int4',
            'L': L.half(),
            'R': R.half(),
            'residual_q': residual_q,
            'scale': scale,
            'offset': offset,
            'shape': list(weight.shape),
        }
    
    def _compress_vq_int2(self, weight: torch.Tensor, alloc: LayerAllocation) -> Dict:
        """Vector quantization + INT2 residual."""
        cb = self.calibrator.codebooks[alloc.codebook_id]
        vec_dim = cb.shape[1]
        
        # Flatten and pad
        flat = weight.flatten()
        orig_len = len(flat)
        pad_len = (vec_dim - len(flat) % vec_dim) % vec_dim
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])
        
        vectors = flat.view(-1, vec_dim)
        
        # Find nearest codebook entries
        dists = torch.cdist(vectors, cb)
        indices = dists.argmin(dim=1)
        
        # Reconstruct and compute residual
        reconstructed = cb[indices]
        residual = vectors - reconstructed
        
        # INT2 quantize residual
        residual_flat = residual.flatten()[:orig_len]
        residual_q, scale, offset = self._int2_quantize(residual_flat, alloc.group_size)
        
        return {
            'method': 'vq_int2',
            'indices': indices.to(torch.uint8 if cb.shape[0] <= 256 else torch.int16),
            'residual_q': residual_q,
            'scale': scale,
            'offset': offset,
            'codebook_id': alloc.codebook_id,
            'shape': list(weight.shape),
            'orig_len': orig_len,
        }
    
    def _compress_vq_only(self, weight: torch.Tensor, alloc: LayerAllocation) -> Dict:
        """Vector quantization only (maximum compression)."""
        cb = self.calibrator.codebooks[alloc.codebook_id]
        vec_dim = cb.shape[1]
        
        # Flatten and pad
        flat = weight.flatten()
        orig_len = len(flat)
        pad_len = (vec_dim - len(flat) % vec_dim) % vec_dim
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])
        
        vectors = flat.view(-1, vec_dim)
        
        # Find nearest codebook entries
        dists = torch.cdist(vectors, cb)
        indices = dists.argmin(dim=1)
        
        return {
            'method': 'vq_only',
            'indices': indices.to(torch.uint8 if cb.shape[0] <= 256 else torch.int16),
            'codebook_id': alloc.codebook_id,
            'shape': list(weight.shape),
            'orig_len': orig_len,
        }
    
    def decompress_layer(self, compressed: Dict) -> torch.Tensor:
        """Decompress a single layer."""
        method = compressed['method']
        shape = compressed['shape']
        
        if method == 'lowrank_int4':
            return self._decompress_lowrank_int4(compressed, shape)
        elif method == 'vq_int2':
            return self._decompress_vq_int2(compressed, shape)
        else:  # vq_only
            return self._decompress_vq_only(compressed, shape)
    
    def _decompress_lowrank_int4(self, compressed: Dict, shape: List[int]) -> torch.Tensor:
        """Decompress low-rank + INT4."""
        L = compressed['L'].float()
        R = compressed['R'].float()
        
        # Low-rank reconstruction
        approx = L @ R
        
        # Dequantize residual
        residual = self._int4_dequantize(
            compressed['residual_q'],
            compressed['scale'],
            compressed['offset'],
            shape
        )
        
        return approx + residual
    
    def _decompress_vq_int2(self, compressed: Dict, shape: List[int]) -> torch.Tensor:
        """Decompress VQ + INT2."""
        cb = self.calibrator.codebooks[compressed['codebook_id']]
        indices = compressed['indices'].long()
        
        # VQ reconstruction
        vectors = cb[indices]
        flat = vectors.flatten()[:compressed['orig_len']]
        
        # Dequantize residual
        residual = self._int2_dequantize(
            compressed['residual_q'],
            compressed['scale'],
            compressed['offset'],
            compressed['orig_len']
        )
        
        return (flat + residual).view(shape)
    
    def _decompress_vq_only(self, compressed: Dict, shape: List[int]) -> torch.Tensor:
        """Decompress VQ only."""
        cb = self.calibrator.codebooks[compressed['codebook_id']]
        indices = compressed['indices'].long()
        
        vectors = cb[indices]
        flat = vectors.flatten()[:compressed['orig_len']]
        
        return flat.view(shape)
    
    def _int4_quantize(self, tensor: torch.Tensor, group_size: int) -> Tuple:
        """INT4 quantization with iterative refinement."""
        flat = tensor.flatten()
        n = len(flat)
        
        # Pad to multiple of group_size
        pad_len = (group_size - n % group_size) % group_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])
        
        num_groups = len(flat) // group_size
        groups = flat.view(num_groups, group_size)
        
        # Compute scales with iterative refinement
        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
        
        for _ in range(5):
            scale = torch.where(
                (g_max - g_min).abs() > 1e-8,
                (g_max - g_min) / 15.0,
                torch.ones_like(g_max)
            )
            inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
            
            q = ((groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 15)
            deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
            err = groups - deq
            
            g_min = g_min + err.min(dim=1).values * 0.5
            g_max = g_max + err.max(dim=1).values * 0.5
        
        # Final quantization
        scale = torch.where(
            (g_max - g_min).abs() > 1e-8,
            (g_max - g_min) / 15.0,
            torch.ones_like(g_max)
        )
        inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
        q = ((groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 15)
        
        # Pack to bytes (2 values per byte)
        q_flat = q.flatten().to(torch.uint8)
        packed = torch.zeros((len(q_flat) + 1) // 2, dtype=torch.uint8, device=flat.device)
        packed = (q_flat[::2] & 0x0f) | ((q_flat[1::2] & 0x0f) << 4) if len(q_flat) > 1 else q_flat
        
        return packed, scale.half(), g_min.half()
    
    def _int4_dequantize(self, packed, scale, offset, shape) -> torch.Tensor:
        """INT4 dequantization."""
        scale = scale.float()
        offset = offset.float()
        
        # Unpack
        low = packed & 0x0f
        high = (packed >> 4) & 0x0f
        q = torch.stack([low, high], dim=1).flatten().float()
        
        # Dequantize
        n = np.prod(shape)
        group_size = len(q) // len(scale)
        q = q[:n].view(-1, group_size)
        
        deq = q * scale.unsqueeze(1) + offset.unsqueeze(1)
        return deq.flatten()[:n].view(shape)
    
    def _int2_quantize(self, tensor: torch.Tensor, group_size: int) -> Tuple:
        """INT2 quantization (4 levels: 0, 1, 2, 3)."""
        flat = tensor.flatten()
        n = len(flat)
        
        pad_len = (group_size - n % group_size) % group_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])
        
        num_groups = len(flat) // group_size
        groups = flat.view(num_groups, group_size)
        
        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
        
        scale = torch.where(
            (g_max - g_min).abs() > 1e-8,
            (g_max - g_min) / 3.0,
            torch.ones_like(g_max)
        )
        inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
        
        q = ((groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 3)
        
        # Pack (4 values per byte)
        q_flat = q.flatten().to(torch.uint8)
        packed_len = (len(q_flat) + 3) // 4
        packed = torch.zeros(packed_len, dtype=torch.uint8, device=flat.device)
        
        for i in range(4):
            if i < len(q_flat[i::4]):
                packed[:len(q_flat[i::4])] |= (q_flat[i::4] & 0x03) << (i * 2)
        
        return packed, scale.half(), g_min.half()
    
    def _int2_dequantize(self, packed, scale, offset, orig_len) -> torch.Tensor:
        """INT2 dequantization."""
        scale = scale.float()
        offset = offset.float()
        
        # Unpack (4 values per byte)
        q0 = packed & 0x03
        q1 = (packed >> 2) & 0x03
        q2 = (packed >> 4) & 0x03
        q3 = (packed >> 6) & 0x03
        q = torch.stack([q0, q1, q2, q3], dim=1).flatten().float()
        
        # Dequantize
        group_size = len(q) // len(scale)
        q = q[:orig_len].view(-1, group_size)
        
        deq = q * scale.unsqueeze(1) + offset.unsqueeze(1)
        return deq.flatten()[:orig_len]


def compress_model_tenpak10x(
    model,
    tokenizer,
    calibration_texts: List[str],
    num_fisher_samples: int = 128,
    num_codebook_epochs: int = 50,
    device: str = 'cuda'
) -> Tuple[Dict, TenPak10XCalibrator]:
    """Full TenPak-10X compression pipeline."""
    
    # Phase 1: Calibration
    calibrator = TenPak10XCalibrator(model, tokenizer, device)
    
    # Collect Fisher information
    calibrator.collect_fisher_info(calibration_texts, num_fisher_samples)
    
    # Allocate bits
    calibrator.allocate_bits(target_avg_bits=3.2)
    
    # Learn shared codebooks
    calibrator.learn_shared_codebooks(calibration_texts, num_codebook_epochs)
    
    # Phase 2: Compression
    compressor = TenPak10XCompressor(calibrator)
    compressed_layers = {}
    
    print("[COMPRESS] Compressing layers...")
    for name, alloc in tqdm(calibrator.allocations.items(), desc="Compressing"):
        # Get weight
        for n, m in model.named_modules():
            if n == name and hasattr(m, 'weight'):
                compressed = compressor.compress_layer(m.weight.data, alloc)
                compressed_layers[name] = compressed
                break
    
    return compressed_layers, calibrator


def decompress_and_load(
    model,
    compressed_layers: Dict,
    calibrator: TenPak10XCalibrator
) -> None:
    """Load compressed weights back into model."""
    
    compressor = TenPak10XCompressor(calibrator)
    
    print("[DECOMPRESS] Loading weights...")
    for name, compressed in tqdm(compressed_layers.items(), desc="Loading"):
        weight = compressor.decompress_layer(compressed)
        
        # Find and update module
        for n, m in model.named_modules():
            if n == name and hasattr(m, 'weight'):
                m.weight.data = weight.to(m.weight.dtype).to(m.weight.device)
                break


if __name__ == "__main__":
    # Test on GPT-2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Sample calibration texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ] * 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Compress
    compressed, calibrator = compress_model_tenpak10x(
        model, tokenizer, texts,
        num_fisher_samples=32,
        num_codebook_epochs=20,
        device=device
    )
    
    # Calculate compression
    original_size = sum(p.numel() * 4 for p in model.parameters())
    print(f"Original size: {original_size / 1e6:.1f} MB")
    
    # Decompress and load
    decompress_and_load(model, compressed, calibrator)
    
    print("Done!")
