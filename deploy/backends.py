from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
        }


class DeploymentBackend:
    backend_id: str
    engine: str
    display_name: str

    def availability(self, *, hardware: str) -> BackendAvailability:
        return BackendAvailability(True)

    def recipe_fragment(self, *, model_id: str, hardware: str) -> Dict[str, Any]:
        raise NotImplementedError


class _TgiBitsandbytes4BitBackend(DeploymentBackend):
    def __init__(self, *, quantize: str):
        self.backend_id = f"tgi.{quantize}"
        self.engine = "tgi"
        self.display_name = f"TGI ({quantize})"
        self._quantize = quantize

    def availability(self, *, hardware: str) -> BackendAvailability:
        if hardware == "cpu":
            return BackendAvailability(False, "bitsandbytes 4-bit requires a CUDA GPU")
        return BackendAvailability(True)

    def recipe_fragment(self, *, model_id: str, hardware: str) -> Dict[str, Any]:
        return {
            "engine": "tgi",
            "quantize": self._quantize,
            "launcher_args": [
                "--model-id",
                model_id,
                "--quantize",
                self._quantize,
            ],
        }


class _TgiStubBackend(DeploymentBackend):
    def __init__(self, *, backend_id: str, display_name: str, quantize: Optional[str], reason: str):
        self.backend_id = backend_id
        self.engine = "tgi"
        self.display_name = display_name
        self._quantize = quantize
        self._reason = reason

    def availability(self, *, hardware: str) -> BackendAvailability:
        return BackendAvailability(False, self._reason)

    def recipe_fragment(self, *, model_id: str, hardware: str) -> Dict[str, Any]:
        frag: Dict[str, Any] = {
            "engine": "tgi",
            "quantize": self._quantize,
        }
        if self._quantize:
            frag["launcher_args"] = [
                "--model-id",
                model_id,
                "--quantize",
                self._quantize,
            ]
        else:
            frag["launcher_args"] = ["--model-id", model_id]
        frag["notes"] = self._reason
        return frag


_BACKENDS: Dict[str, DeploymentBackend] = {
    "tgi.bitsandbytes-nf4": _TgiBitsandbytes4BitBackend(quantize="bitsandbytes-nf4"),
    "tgi.bitsandbytes-fp4": _TgiBitsandbytes4BitBackend(quantize="bitsandbytes-fp4"),
    "tgi.awq": _TgiStubBackend(
        backend_id="tgi.awq",
        display_name="TGI (AWQ)",
        quantize="awq",
        reason="Stub: requires pre-quantized AWQ weights; TenPak export not implemented yet",
    ),
    "tgi.gptq": _TgiStubBackend(
        backend_id="tgi.gptq",
        display_name="TGI (GPTQ)",
        quantize="gptq",
        reason="Stub: requires pre-quantized GPTQ weights; TenPak export not implemented yet",
    ),
    "tgi.quanto": _TgiStubBackend(
        backend_id="tgi.quanto",
        display_name="TGI (Quanto)",
        quantize=None,
        reason="Stub: Quanto export/recipe not implemented yet",
    ),
}


def list_backend_ids() -> List[str]:
    return sorted(_BACKENDS.keys())


def list_backends() -> List[DeploymentBackend]:
    return [get_backend(bid) for bid in list_backend_ids()]


def get_backend(backend_id: str) -> DeploymentBackend:
    if backend_id not in _BACKENDS:
        raise KeyError(f"Unknown backend_id: {backend_id}. Available: {', '.join(list_backend_ids())}")
    return _BACKENDS[backend_id]


def default_backend_for_engine(engine: str) -> str:
    if engine == "tgi":
        return "tgi.bitsandbytes-nf4"
    raise KeyError(f"No default backend for engine: {engine}")
