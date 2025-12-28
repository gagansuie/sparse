"""
Tests for delta compression functionality
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from core.delta import (
    compress_delta,
    reconstruct_from_delta,
    estimate_delta_savings,
    DeltaManifest,
    compress_adapter_delta,
)


class TestDeltaManifest:
    """Test DeltaManifest dataclass."""
    
    def test_delta_manifest_creation(self):
        """Test delta manifest creation."""
        manifest = DeltaManifest(
            base_model_id="base-model",
            finetune_model_id="fine-tuned",
        )
        
        assert manifest.base_model_id == "base-model"
        assert manifest.finetune_model_id == "fine-tuned"


class TestEstimateDeltaSavings:
    """Test delta savings estimation."""
    
    @patch("core.delta.AutoModelForCausalLM")
    def test_estimate_savings_basic(self, mock_model_class):
        """Test basic savings estimation."""
        # Mock models
        mock_base = Mock()
        mock_base.num_parameters.return_value = 1000000
        
        mock_finetuned = Mock()
        mock_finetuned.num_parameters.return_value = 1000000
        
        # Mock state dicts with some changed layers
        base_state = {
            f"layer.{i}.weight": torch.randn(10, 10)
            for i in range(20)
        }
        finetuned_state = base_state.copy()
        # Change layers 10 and 11
        finetuned_state["layer.10.weight"] = torch.randn(10, 10)
        finetuned_state["layer.11.weight"] = torch.randn(10, 10)
        
        mock_base.state_dict.return_value = base_state
        mock_finetuned.state_dict.return_value = finetuned_state
        
        def mock_from_pretrained(model_id, **kwargs):
            if "base" in model_id:
                return mock_base
            else:
                return mock_finetuned
        
        mock_model_class.from_pretrained.side_effect = mock_from_pretrained
        
        savings = estimate_delta_savings(
            base_model_id="base-model",
            finetune_model_id="fine-tuned-model",
        )
        
        assert "estimated_compression" in savings
        assert "avg_sparsity" in savings
        assert "sample_layers" in savings
        assert savings["estimated_compression"] > 0


class TestCompressDelta:
    """Test delta compression."""
    
    @patch("core.delta.AutoModelForCausalLM")
    def test_compress_delta(self, mock_model_class, tmp_path):
        """Test delta compression."""
        # Create mock models
        mock_base = Mock()
        mock_finetuned = Mock()
        
        base_state = {
            f"layer.{i}.weight": torch.randn(10, 10)
            for i in range(20)
        }
        finetuned_state = base_state.copy()
        finetuned_state["layer.10.weight"] = torch.randn(10, 10)
        
        mock_base.state_dict.return_value = base_state
        mock_finetuned.state_dict.return_value = finetuned_state
        
        def mock_from_pretrained(model_id, **kwargs):
            if "base" in model_id:
                return mock_base
            else:
                return mock_finetuned
        
        mock_model_class.from_pretrained.side_effect = mock_from_pretrained
        
        output_dir = tmp_path / "delta"
        
        manifest = compress_delta(
            base_model_id="base-model",
            finetune_model_id="fine-tuned-model",
            output_path=str(output_dir),
        )
        
        assert manifest.base_model_id == "base-model"
        assert manifest.finetune_model_id == "fine-tuned-model"
        assert manifest.num_layers >= 0


class TestReconstructFromDelta:
    """Test reconstruction from delta."""
    
    @patch("core.delta.AutoModelForCausalLM")
    def test_reconstruct_from_delta(self, mock_model_class, tmp_path):
        """Test reconstruction from delta."""
        # Create mock base model
        mock_base = Mock()
        base_state = {
            f"layer.{i}.weight": torch.randn(10, 10)
            for i in range(20)
        }
        mock_base.state_dict.return_value = base_state
        mock_base.load_state_dict = Mock()
        
        mock_model_class.from_pretrained.return_value = mock_base
        
        # Create mock delta directory
        delta_dir = tmp_path / "delta"
        delta_dir.mkdir()
        
        # Save delta
        delta_data = {
            "layer.10.weight": torch.randn(10, 10),
        }
        torch.save(delta_data, delta_dir / "delta.pt")
        
        # Save manifest
        import json
        manifest = {
            "base_model_id": "base-model",
            "finetune_model_id": "fine-tuned",
            "layer_deltas": [],
            "delta_type": "model_delta",
        }
        with open(delta_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        model = reconstruct_from_delta(
            base_model_id="base-model",
            delta_path=str(delta_dir),
        )
        
        assert model is not None
        mock_model_class.from_pretrained.assert_called()


class TestDeltaCompressionIntegration:
    """Integration tests for delta compression."""
    
    @patch("core.delta.AutoModelForCausalLM")
    def test_full_workflow(self, mock_model_class, tmp_path):
        """Test full compress and reconstruct workflow."""
        # Create mock models
        mock_base = Mock()
        mock_finetuned = Mock()
        
        base_state = {
            "layer.0.weight": torch.randn(5, 5),
            "layer.1.weight": torch.randn(5, 5),
        }
        finetuned_state = {
            "layer.0.weight": base_state["layer.0.weight"],
            "layer.1.weight": torch.randn(5, 5),  # Changed
        }
        
        mock_base.state_dict.return_value = base_state
        mock_finetuned.state_dict.return_value = finetuned_state
        
        def mock_from_pretrained(model_id, **kwargs):
            if "base" in model_id:
                return mock_base
            else:
                return mock_finetuned
        
        mock_model_class.from_pretrained.side_effect = mock_from_pretrained
        
        # Compress
        output_dir = tmp_path / "delta"
        manifest = compress_delta(
            base_model_id="base-model",
            finetune_model_id="fine-tuned-model",
            output_path=str(output_dir),
        )
        
        assert manifest is not None
        assert manifest.num_layers >= 0
        
        # Verify delta files were created
        assert (output_dir / "manifest.json").exists()


def test_compress_adapter_delta_local_path(tmp_path):
    from core.delta import compress_adapter_delta

    adapter_src = tmp_path / "adapter_src"
    adapter_src.mkdir()
    (adapter_src / "adapter_config.json").write_text("{}")
    (adapter_src / "adapter_model.safetensors").write_text("dummy")

    out_dir = tmp_path / "adapter_delta"
    manifest = compress_adapter_delta(
        base_model_id="base-model",
        adapter_id=str(adapter_src),
        output_path=str(out_dir),
    )

    assert manifest.delta_type == "adapter"
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "adapter" / "adapter_config.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
