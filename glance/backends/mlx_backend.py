"""Optional Apple-silicon fast path via mlx-vlm (`pip install mlx-vlm`).

Status: on an 8 GB M1 this is not optional -- it is the only way the model fits.
Qwen2.5-VL-3B is ~7.5 GB in bf16, which will not load alongside macOS in 8 GB of
unified memory. The 4-bit MLX build is ~2.1 GB resident.

Parameter accounting: MLX community checkpoints are quantised and bit-packed, so
shape-based counting of *those* files is misleading. `glance budget` therefore
always audits the original HF repo (spec.hf_id). Quantisation changes bytes per
weight, not the number of learned weights, so the 3.75B total stands either way.

Known gap: no logit access is wired up here, so yes_probability() falls back to
the base class's parse-the-text version and returns only 0.0 or 1.0. Confidence
from this backend is binary, not calibrated. The HF backend is the reference for
any claim about confidence quality.
"""
from __future__ import annotations

import os
import tempfile
import time

from ..config import ModelSpec
from .base import Backend, GenResult


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, spec: ModelSpec, model_id: str | None = None):
        super().__init__(spec)
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as e:
            raise RuntimeError(
                "MLX backend needs: pip install mlx-vlm  (Apple silicon only)"
            ) from e
        self._generate_fn = generate
        self._template = apply_chat_template
        self.model_id = model_id or spec.mlx_id or spec.hf_id
        self.model, self.processor = load(self.model_id)
        self._config = getattr(self.model, "config", None)

    @property
    def device(self) -> str:
        return "mlx-gpu"

    def _generate(self, image, prompt, max_new_tokens, system):
        full = f"{system}\n\n{prompt}" if system else prompt
        formatted = self._template(self.processor, self._config, full, num_images=1)
        # mlx-vlm takes image paths, so round-trip through a temp PNG.
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.convert("RGB").save(path)
            t0 = time.perf_counter()
            out = self._generate_fn(self.model, self.processor, formatted, [path],
                                    max_tokens=max_new_tokens, verbose=False,
                                    temperature=0.0)
            dt = time.perf_counter() - t0
        finally:
            os.unlink(path)
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        n_new = int(getattr(out, "generation_tokens", 0) or 0)
        n_in = int(getattr(out, "prompt_tokens", 0) or 0)
        return GenResult(text=text.strip(), prompt_tokens=n_in,
                         new_tokens=n_new or len(text.split()), latency_s=dt)

    def describe(self) -> dict:
        d = super().describe()
        d["mlx_checkpoint"] = self.model_id
        d["quantised"] = "4bit" in self.model_id.lower()
        d["logit_access"] = False
        return d