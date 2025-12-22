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
)


class TestDeltaManifest:
    """Test DeltaManifest dataclass."""
    
    def test_delta_manifest_creation(self):
        """Test delta manifest creation."""
        manifest = DeltaManifest(
            base_model_id="base-model",
            finetuned_model_id="fine-tuned",
            changed_layers=["layer.10", "layer.11"],
            delta_method="sparse_int8",
        )
        
        assert manifest.base_model_id == "base-model"
        assert manifest.finetuned_model_id == "fine-tuned"
        assert len(manifest.changed_layers) == 2
        assert manifest.delta_method == "sparse_int8"


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
            finetuned_model_id="fine-tuned-model",
        )
        
        assert "base_size_gb" in savings
        assert "finetuned_size_gb" in savings
        assert "delta_size_gb" in savings
        assert "savings_pct" in savings
        assert savings["savings_pct"] > 0
        assert savings["delta_size_gb"] < savings["finetuned_size_gb"]


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
            finetuned_model_id="fine-tuned-model",
            output_dir=str(output_dir),
        )
        
        assert manifest.base_model_id == "base-model"
        assert manifest.finetuned_model_id == "fine-tuned-model"
        assert len(manifest.changed_layers) > 0


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
            "finetuned_model_id": "fine-tuned",
            "changed_layers": ["layer.10.weight"],
            "delta_method": "sparse_int8",
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
            finetuned_model_id="fine-tuned-model",
            output_dir=str(output_dir),
        )
        
        assert manifest is not None
        assert len(manifest.changed_layers) > 0
        
        # Verify delta files were created
        assert (output_dir / "manifest.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
