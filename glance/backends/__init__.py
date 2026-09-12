"""Backend factory."""
from __future__ import annotations

from ..config import DEFAULT_MODEL, ModelSpec, get_spec
from .base import Backend, GenResult

__all__ = ["Backend", "GenResult", "build_backend", "available_backends"]


def available_backends() -> list[str]:
    out = ["hf"]
    try:
        import mlx_vlm  # noqa: F401
        out.append("mlx")
    except ImportError:
        pass
    return out


def build_backend(model: str | ModelSpec = DEFAULT_MODEL, backend: str = "auto",
                  device: str = "auto", dtype: str = "auto",
                  local_files_only: bool = False) -> Backend:
    spec = model if isinstance(model, ModelSpec) else get_spec(model)

    if backend == "auto":
        # Prefer MLX only when it is installed AND this model has an MLX build.
        backend = "mlx" if ("mlx" in available_backends() and spec.mlx_id) else "hf"

    if backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend(spec)
    if backend == "hf":
        from .hf import HFBackend
        return HFBackend(spec, device=device, dtype=dtype, local_files_only=local_files_only)
    raise ValueError(f"Unknown backend '{backend}'. Available: {', '.join(available_backends())}")