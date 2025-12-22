"""
TenPak Quantization Wrapper

Wraps industry-standard quantization tools (AutoGPTQ, AutoAWQ, bitsandbytes)
instead of implementing custom codecs.

TenPak focuses on:
- Delta compression for fine-tunes
- Streaming artifact format
- Cost optimization (benchmarking different methods)
- Enterprise features (signing, verification)
"""

from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
import torch

QuantizationMethod = Literal["gptq", "awq", "bitsandbytes", "none"]


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    
    # GPTQ-specific
    desc_act: bool = False
    sym: bool = True
    
    # AWQ-specific
    zero_point: bool = True
    
    # bitsandbytes-specific
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False


class QuantizationWrapper:
    """Wrapper for industry-standard quantization tools."""
    
    @staticmethod
    def quantize_model(
        model_id: str,
        config: QuantizationConfig,
        calibration_data: Optional[Any] = None,
        device: str = "cuda",
    ) -> "torch.nn.Module":
        """
        Quantize a model using the specified method.
        
        Args:
            model_id: HuggingFace model ID
            config: Quantization configuration
            calibration_data: Optional calibration dataset
            device: Target device
            
        Returns:
            Quantized model
        """
        if config.method == "gptq":
            return QuantizationWrapper._quantize_gptq(
                model_id, config, calibration_data, device
            )
        elif config.method == "awq":
            return QuantizationWrapper._quantize_awq(
                model_id, config, calibration_data, device
            )
        elif config.method == "bitsandbytes":
            return QuantizationWrapper._quantize_bitsandbytes(
                model_id, config, device
            )
        elif config.method == "none":
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
            )
        else:
            raise ValueError(f"Unknown quantization method: {config.method}")
    
    @staticmethod
    def _quantize_gptq(
        model_id: str,
        config: QuantizationConfig,
        calibration_data: Optional[Any],
        device: str,
    ) -> "torch.nn.Module":
        """Quantize using AutoGPTQ."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "AutoGPTQ not installed. Install with: pip install auto-gptq"
            )
        
        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            desc_act=config.desc_act,
            sym=config.sym,
        )
        
        model = AutoGPTQForCausalLM.from_pretrained(
            model_id,
            quantize_config=quantize_config,
            device_map=device,
        )
        
        if calibration_data is not None:
            model.quantize(calibration_data)
        
        return model
    
    @staticmethod
    def _quantize_awq(
        model_id: str,
        config: QuantizationConfig,
        calibration_data: Optional[Any],
        device: str,
    ) -> "torch.nn.Module":
        """Quantize using AutoAWQ."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "AutoAWQ not installed. Install with: pip install autoawq"
            )
        
        model = AutoAWQForCausalLM.from_pretrained(model_id)
        
        if calibration_data is not None:
            from awq import AutoAWQConfig
            
            quant_config = AutoAWQConfig(
                w_bit=config.bits,
                q_group_size=config.group_size,
                zero_point=config.zero_point,
            )
            
            model.quantize(
                tokenizer=None,  # Will be loaded internally
                quant_config=quant_config,
                calib_data=calibration_data,
            )
        
        return model
    
    @staticmethod
    def _quantize_bitsandbytes(
        model_id: str,
        config: QuantizationConfig,
        device: str,
    ) -> "torch.nn.Module":
        """Quantize using bitsandbytes."""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "bitsandbytes not installed. Install with: pip install bitsandbytes"
            )
        
        if config.bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=config.llm_int8_threshold,
                llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            )
        elif config.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError(f"bitsandbytes only supports 4-bit or 8-bit, got {config.bits}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
        )
        
        return model
    
    @staticmethod
    def estimate_size(
        model_id: str,
        config: QuantizationConfig,
    ) -> Dict[str, float]:
        """
        Estimate model size after quantization.
        
        Returns:
            Dict with original_size_gb, quantized_size_gb, compression_ratio
        """
        from transformers import AutoConfig
        
        model_config = AutoConfig.from_pretrained(model_id)
        
        # Estimate parameter count
        if hasattr(model_config, "num_parameters"):
            num_params = model_config.num_parameters
        else:
            # Rough estimate for transformer models
            hidden_size = model_config.hidden_size
            num_layers = model_config.num_hidden_layers
            vocab_size = model_config.vocab_size
            
            # Attention + MLP per layer + embeddings
            params_per_layer = (
                4 * hidden_size * hidden_size +  # QKV + O
                8 * hidden_size * hidden_size    # MLP (2 * 4x expansion)
            )
            
            num_params = (
                num_layers * params_per_layer +
                vocab_size * hidden_size * 2  # Input + output embeddings
            )
        
        # Original size (FP16)
        original_size_gb = (num_params * 2) / (1024 ** 3)
        
        # Quantized size
        if config.method == "none":
            quantized_size_gb = original_size_gb
        elif config.bits == 8:
            quantized_size_gb = (num_params * 1) / (1024 ** 3)
        elif config.bits == 4:
            # 4-bit + scales/zeros overhead
            quantized_size_gb = (num_params * 0.5 * 1.1) / (1024 ** 3)
        else:
            quantized_size_gb = (num_params * config.bits / 8) / (1024 ** 3)
        
        compression_ratio = original_size_gb / quantized_size_gb if quantized_size_gb > 0 else 1.0
        
        return {
            "original_size_gb": original_size_gb,
            "quantized_size_gb": quantized_size_gb,
            "compression_ratio": compression_ratio,
        }


# Predefined configurations for common use cases
QUANTIZATION_PRESETS = {
    "gptq_quality": QuantizationConfig(
        method="gptq",
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
    ),
    "gptq_balanced": QuantizationConfig(
        method="gptq",
        bits=4,
        group_size=256,
        desc_act=False,
        sym=True,
    ),
    "gptq_size": QuantizationConfig(
        method="gptq",
        bits=4,
        group_size=512,
        desc_act=False,
        sym=True,
    ),
    "awq_quality": QuantizationConfig(
        method="awq",
        bits=4,
        group_size=128,
        zero_point=True,
    ),
    "awq_balanced": QuantizationConfig(
        method="awq",
        bits=4,
        group_size=256,
        zero_point=True,
    ),
    "bnb_int8": QuantizationConfig(
        method="bitsandbytes",
        bits=8,
        llm_int8_threshold=6.0,
    ),
    "bnb_nf4": QuantizationConfig(
        method="bitsandbytes",
        bits=4,
    ),
    "fp16": QuantizationConfig(
        method="none",
        bits=16,
    ),
}
