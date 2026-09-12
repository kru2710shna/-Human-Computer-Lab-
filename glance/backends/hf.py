"""Local inference with Hugging Face transformers on Apple MPS, CUDA, or CPU."""
from __future__ import annotations

import time

from PIL import Image

from ..config import ModelSpec
from .base import Backend, GenResult


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str, requested: str):
    import torch

    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if requested != "auto":
        return table[requested]
    if device == "cpu":
        # CPU bf16 matmul is slow-pathed on many builds; fp32 is faster there.
        return torch.float32
    # bf16 halves memory vs fp32 and avoids the fp16 overflows some ViTs hit.
    return torch.bfloat16


def _sync(device: str) -> None:
    """Timings are meaningless without this: MPS and CUDA queue work async."""
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class HFBackend(Backend):
    name = "hf"

    def __init__(self, spec: ModelSpec, device: str = "auto", dtype: str = "auto",
                 local_files_only: bool = False, attn: str | None = None):
        super().__init__(spec)
        import torch
        import transformers
        from transformers import AutoProcessor

        self._torch = torch
        self._device = _pick_device(device)
        self._dtype = _pick_dtype(self._device, dtype)

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        # transformers renamed torch_dtype -> dtype in 4.56. The wrong keyword is
        # silently swallowed into **kwargs and you get a fp32 model without warning,
        # so pick the right one rather than passing both.
        dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        kw = {dkey: self._dtype, "local_files_only": local_files_only}
        if attn:
            kw["attn_implementation"] = attn

        try:
            from transformers import AutoModelForImageTextToText as Auto
            self.model = Auto.from_pretrained(spec.hf_id, **kw)
        except (ImportError, ValueError):
            from transformers import Qwen2_5_VLForConditionalGeneration as Auto
            self.model = Auto.from_pretrained(spec.hf_id, **kw)

        self.model.to(self._device).eval()

        # Pinning min/max pixels here keeps the processor's own resize a no-op,
        # because imaging.prepare() has already snapped the image to a legal size.
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_id, local_files_only=local_files_only,
            min_pixels=spec.min_pixels, max_pixels=spec.max_pixels,
        )

        tok = self.processor.tokenizer
        self._yes_ids = self._first_ids(tok, ("Yes", "yes", " Yes", " yes", "YES"))
        self._no_ids = self._first_ids(tok, ("No", "no", " No", " no", "NO"))

    @staticmethod
    def _first_ids(tok, words) -> list[int]:
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    @property
    def device(self) -> str:
        return self._device

    def _inputs(self, image: Image.Image, prompt: str, system: str | None):
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        return inputs.to(self._device)

    def _generate(self, image, prompt, max_new_tokens, system):
        torch = self._torch
        inputs = self._inputs(image, prompt, system)
        n_in = int(inputs["input_ids"].shape[1])
        _sync(self._device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        _sync(self._device)
        dt = time.perf_counter() - t0
        new = out[:, n_in:]
        text = self.processor.batch_decode(new, skip_special_tokens=True)[0]
        return GenResult(text=text.strip(), prompt_tokens=n_in,
                         new_tokens=int(new.shape[1]), latency_s=dt)

    def yes_probability(self, image, question):
        """One forward pass; read the next-token distribution over Yes/No.

        Renormalising over just the Yes and No token groups discards probability
        the model put on anything else, which is what we want: we asked a binary
        question and we want its belief *given* that it answers it.
        """
        torch = self._torch
        inputs = self._inputs(image, question + " Answer with Yes or No only.", None)
        _sync(self._device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                logits = self.model(**inputs, logits_to_keep=1).logits[0, -1]
            except TypeError:
                logits = self.model(**inputs).logits[0, -1]
        _sync(self._device)
        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = float(probs[self._yes_ids].sum())
        p_no = float(probs[self._no_ids].sum())
        self.calls.append(GenResult(text=f"p_yes={p_yes:.4f} p_no={p_no:.4f}",
                                    prompt_tokens=int(inputs["input_ids"].shape[1]),
                                    new_tokens=1, latency_s=time.perf_counter() - t0))
        if p_yes + p_no < 1e-6:
            return None  # model wants to say something else entirely; abstain
        return p_yes / (p_yes + p_no)

    def runtime_param_count(self) -> int:
        # parameters() de-duplicates tied weights, matching the checkpoint count.
        return sum(p.numel() for p in self.model.parameters())

    def memory_footprint(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) + \
               sum(b.numel() * b.element_size() for b in self.model.buffers())

    def describe(self) -> dict:
        d = super().describe()
        d["dtype"] = str(self._dtype).replace("torch.", "")
        d["weights_bytes"] = self.memory_footprint()
        return d