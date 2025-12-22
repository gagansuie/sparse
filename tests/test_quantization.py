"""
Tests for quantization wrapper functionality
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from core.quantization import (
    QuantizationWrapper,
    QuantizationConfig,
    QuantizationMethod,
    QUANTIZATION_PRESETS,
)


class TestQuantizationConfig:
    """Test QuantizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = QuantizationConfig(method="gptq")
        assert config.method == "gptq"
        assert config.bits == 4
        assert config.group_size == 128
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantizationConfig(
            method="awq",
            bits=8,
            group_size=256,
            zero_point=False,
        )
        assert config.method == "awq"
        assert config.bits == 8
        assert config.group_size == 256
        assert config.zero_point is False


class TestQuantizationPresets:
    """Test predefined quantization presets."""
    
    def test_all_presets_exist(self):
        """Test that all expected presets exist."""
        expected_presets = [
            "fp16",
            "gptq_quality",
            "gptq_balanced",
            "gptq_size",
            "awq_quality",
            "awq_balanced",
            "bnb_int8",
            "bnb_nf4",
        ]
        
        for preset in expected_presets:
            assert preset in QUANTIZATION_PRESETS, f"Missing preset: {preset}"
    
    def test_gptq_presets(self):
        """Test GPTQ preset configurations."""
        gptq_quality = QUANTIZATION_PRESETS["gptq_quality"]
        assert gptq_quality.method == QuantizationMethod.GPTQ
        assert gptq_quality.bits == 4
        assert gptq_quality.group_size == 128
        assert gptq_quality.requires_calibration is True
    
    def test_awq_presets(self):
        """Test AWQ preset configurations."""
        awq_balanced = QUANTIZATION_PRESETS["awq_balanced"]
        assert awq_balanced.method == QuantizationMethod.AWQ
        assert awq_balanced.bits == 4
        assert awq_balanced.group_size == 256
        assert awq_balanced.requires_calibration is True
    
    def test_bnb_presets(self):
        """Test bitsandbytes preset configurations."""
        bnb_nf4 = QUANTIZATION_PRESETS["bnb_nf4"]
        assert bnb_nf4.method == QuantizationMethod.BITSANDBYTES
        assert bnb_nf4.bits == 4
        assert bnb_nf4.requires_calibration is False


class TestQuantizationWrapper:
    """Test QuantizationWrapper functionality."""
    
    def test_estimate_size_gpt2(self):
        """Test size estimation for GPT-2."""
        config = QuantizationConfig(method="gptq", bits=4, group_size=128)
        
        with patch("core.quantization.AutoConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.hidden_size = 768
            mock_config.num_hidden_layers = 12
            mock_config.vocab_size = 50257
            mock_config_class.from_pretrained.return_value = mock_config
            
            size_info = QuantizationWrapper.estimate_size("gpt2", config)
            
            assert "original_size_gb" in size_info
            assert "quantized_size_gb" in size_info
            assert "compression_ratio" in size_info
            assert size_info["compression_ratio"] > 1.0
    
    def test_estimate_size_compression_ratios(self):
        """Test that different bit widths give correct compression ratios."""
        with patch("core.quantization.AutoConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.hidden_size = 768
            mock_config.num_hidden_layers = 12
            mock_config.vocab_size = 50257
            mock_config_class.from_pretrained.return_value = mock_config
            
            # 4-bit should give ~4x compression (accounting for overhead)
            config_4bit = QuantizationConfig(method="gptq", bits=4)
            size_4bit = QuantizationWrapper.estimate_size("gpt2", config_4bit)
            assert 3.0 < size_4bit["compression_ratio"] < 4.5
            
            # 8-bit should give ~2x compression
            config_8bit = QuantizationConfig(method="gptq", bits=8)
            size_8bit = QuantizationWrapper.estimate_size("gpt2", config_8bit)
            assert 1.8 < size_8bit["compression_ratio"] < 2.2
            
            # FP16 (none) should give ~1x
            config_none = QuantizationConfig(method="none")
            size_none = QuantizationWrapper.estimate_size("gpt2", config_none)
            assert size_none["compression_ratio"] == 1.0
    
    @patch("core.quantization.AutoGPTQForCausalLM")
    @patch("core.quantization.BaseQuantizeConfig")
    def test_quantize_gptq(self, mock_quant_config, mock_gptq):
        """Test GPTQ quantization."""
        config = QuantizationConfig(
            method="gptq",
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
        )
        
        mock_model = Mock()
        mock_gptq.from_pretrained.return_value = mock_model
        
        result = QuantizationWrapper.quantize_model(
            model_id="gpt2",
            config=config,
            device="cuda",
        )
        
        assert result == mock_model
        mock_gptq.from_pretrained.assert_called_once()
    
    @patch("core.quantization.AutoAWQForCausalLM")
    def test_quantize_awq(self, mock_awq):
        """Test AWQ quantization."""
        config = QuantizationConfig(
            method="awq",
            bits=4,
            group_size=128,
            zero_point=True,
        )
        
        mock_model = Mock()
        mock_awq.from_pretrained.return_value = mock_model
        
        result = QuantizationWrapper.quantize_model(
            model_id="gpt2",
            config=config,
            device="cuda",
        )
        
        assert result == mock_model
        mock_awq.from_pretrained.assert_called_once()
    
    @patch("core.quantization.AutoModelForCausalLM")
    @patch("core.quantization.BitsAndBytesConfig")
    def test_quantize_bitsandbytes_4bit(self, mock_bnb_config, mock_model):
        """Test bitsandbytes 4-bit quantization."""
        config = QuantizationConfig(method="bitsandbytes", bits=4)
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        result = QuantizationWrapper.quantize_model(
            model_id="gpt2",
            config=config,
            device="cuda",
        )
        
        assert result == mock_model_instance
        mock_model.from_pretrained.assert_called_once()
        mock_bnb_config.assert_called_once()
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        config = QuantizationConfig(method="invalid_method")
        
        with pytest.raises(ValueError, match="Unknown quantization method"):
            QuantizationWrapper.quantize_model(
                model_id="gpt2",
                config=config,
                device="cuda",
            )


class TestQuantizationMethods:
    """Test QuantizationMethod enum."""
    
    def test_all_methods_defined(self):
        """Test that all expected methods are defined."""
        expected_methods = ["GPTQ", "AWQ", "BITSANDBYTES", "FP16"]
        
        for method in expected_methods:
            assert hasattr(QuantizationMethod, method)
    
    def test_method_values(self):
        """Test method string values."""
        assert QuantizationMethod.GPTQ.value == "gptq"
        assert QuantizationMethod.AWQ.value == "awq"
        assert QuantizationMethod.BITSANDBYTES.value == "bitsandbytes"
        assert QuantizationMethod.FP16.value == "fp16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
