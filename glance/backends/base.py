"""Backend interface: a local image+text -> text generator.

Implementations must never call a remote inference service. The challenge
requires the complete inference path to run locally, so anything that would
reach out over the network at generate() time does not belong here.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from ..config import ModelSpec
from ..parsing import parse_yes_no


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    new_tokens: int = 0
    latency_s: float = 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.latency_s if self.latency_s > 0 else 0.0


class Backend(ABC):
    name = "base"

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.calls: list[GenResult] = []

    @abstractmethod
    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int,
                  system: str | None) -> GenResult: ...

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 512,
                 system: str | None = None) -> GenResult:
        t0 = time.perf_counter()
        res = self._generate(image, prompt, max_new_tokens, system)
        if not res.latency_s:
            res.latency_s = time.perf_counter() - t0
        self.calls.append(res)
        return res

    def yes_probability(self, image: Image.Image, question: str) -> float | None:
        """P(yes) for a yes/no question.

        The default implementation generates text and parses it, which only ever
        yields 0.0 or 1.0 -- usable, but not calibrated. Backends with logit
        access override this to return a real probability. Returns None when the
        model does not answer the question at all (an abstention, not a 'no').
        """
        res = self.generate(image, question + " Answer with Yes or No only.", max_new_tokens=4)
        ans = parse_yes_no(res.text)
        return None if ans is None else (1.0 if ans else 0.0)

    def runtime_param_count(self) -> int | None:
        """Live parameter count, cross-checked against the header audit."""
        return None

    @property
    def device(self) -> str:
        return "cpu"

    def stats(self) -> dict:
        """Aggregate cost of every call made through this backend so far."""
        if not self.calls:
            return {"calls": 0}
        gen = [c for c in self.calls if c.new_tokens > 0]
        total_s = sum(c.latency_s for c in self.calls)
        new = sum(c.new_tokens for c in self.calls)
        return {
            "calls": len(self.calls),
            "total_latency_s": round(total_s, 3),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "new_tokens": new,
            "decode_tokens_per_s": round(new / total_s, 2) if total_s else 0.0,
            "mean_call_s": round(total_s / len(self.calls), 3),
            "max_call_s": round(max(c.latency_s for c in self.calls), 3),
        }

    def reset_stats(self) -> None:
        self.calls.clear()

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.spec.hf_id, "device": self.device}