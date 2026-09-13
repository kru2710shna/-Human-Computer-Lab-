# Glance

Fully-local screenshot understanding and UI grounding, under a 6-billion-parameter
budget. Point it at a screen, ask in plain language, get back structured output
including exact pixel coordinates you could click. Nothing leaves the machine.

Built for the Human Computer Lab challenge, Text-and-Vision track.

---

## The constraint, answered first

The challenge caps the complete local inference path at **6B total learned
parameters**, counting every encoder, backbone, adapter, projector, head, and
auxiliary generator. Glance runs **one model** and audits it programmatically
rather than quoting a model card.

    $ glance budget --model qwen2.5-vl-3b

| Component | Parameters |
|---|---|
| Language backbone | 2,774,773,760 |
| Vision encoder | 631,975,680 |
| Token embeddings (output head tied) | 311,164,928 |
| Vision-language projector | 36,708,608 |
| **Total** | **3,754,622,976** |
| Budget | 6,000,000,000 |
| **Headroom** | **2,245,377,024 (37.4% unused)** |

`glance/budget.py` reads the real `safetensors` headers over HTTP range requests
(a few KB, no weights downloaded) and sums tensor shapes itself. It de-duplicates
across shards, and it detects tied embeddings: Qwen2.5-VL-3B sets
`tie_word_embeddings=true`, so its output head shares the input embedding matrix
and is absent from the checkpoint. We count it once and say so in the report,
rather than letting it silently vanish.

Counting from shapes means this is **total** parameters, unaffected by
quantisation or by which weights are active at inference.

**The audit rejects as well as accepts.** A check that has never failed proves
nothing, so an over-budget model is registered as a negative control:

    $ glance budget --model qwen2.5-vl-7b ; echo $?
    ...
    FAIL: 8,292,166,656 / 6,000,000,000 parameters
    1

Non-zero exit on violation, so this works as a CI gate. Note the 7B has
`tie_word_embeddings=false` — its output head is a separate 545M tensor counted
on its own. The audit handles both conventions and reports which applies.

A live cross-check is available too: `glance budget --verify-runtime` loads the
model and compares `sum(p.numel())` against the header total. It reports a
mismatch rather than asserting, because a disagreement would be a real finding
about the checkpoint.

---

## What it does

| Command | Output |
|---|---|
| `glance describe shot.png` | App, screen, visible state, available actions |
| `glance ocr shot.png` | Text with boxes, in reading order |
| `glance find shot.png "the blue Save button"` | Click point + box + confidence |
| `glance guide shot.png "how do I export?"` | Steps, each grounded to a real box |
| `glance diff before.png after.png` | What changed, semantically |
| `glance redact shot.png` | PII located and blurred locally |
| `glance ask shot.png "is anything unsaved?"` | Free-form VQA |
| `glance bench` | Accuracy, latency, memory, reproducible |

Add `--report out.html` to any command for a self-contained report (image
embedded as base64, CSS inlined, no CDN) or `--annotate out.png` for a boxed
screenshot.

---

## How it works

A 3.75B VLM fed a downscaled 1440p screenshot sees UI text a few pixels tall and
guesses. Rather than spend parameters we do not have, Glance spends **compute**:

**1. Zoom-refine.** Ask on the whole screen, get a rough box, then crop generously
around it from the *original* pixels and upscale that crop towards the model's
full pixel budget. A button that was 8px tall is now 60px tall. Ask again inside
the crop.

**2. Logit-level verification.** Instead of asking "is that right?" and trusting
the generated word, read the next-token distribution and renormalise over the
Yes/No token groups. This yields a probability that can be thresholded, and lets
the system abstain rather than point confidently at the wrong thing.

**3. An affine coordinate ledger.** Every derived view carries
`(ox, oy, sx, sy)` mapping its pixels back to the original screenshot. Resize and
crop update it, so a box found in a zoomed crop maps home exactly. This is the
unglamorous piece that makes the other two legal, and it is tested independently
of any model (`tests/test_imaging.py`).

Optional fallbacks: a tile sweep when the first pass finds nothing, and NxN tiled
OCR for small text. Both are opt-in because they cost N² model calls.

---

## Results

Measured on the hardware below, with the 4-bit MLX build. Full JSON in `runs/`.

### Does zoom-refine earn its latency?

This is the question the project hinges on. It roughly doubles latency, so it has
to buy something.

**On buttons and input fields** (elements whose drawn rectangle *is* the element),
6 screens:

| Variant | IoU mean | IoU>=0.50 | IoU>=0.75 | Median latency |
|---|---|---|---|---|
| coarse | 0.536 | 67% | **0%** | 10.6s |
| refine | **0.911** | **100%** | **100%** | 21.0s |

Every one of the six trials improved: 0.58→0.91, 0.54→0.93, 0.71→0.88, 0.27→0.96,
0.47→0.89, 0.64→0.89. Not a single trial above IoU 0.75 without refine; all six
above it with. **+70% IoU for 2x latency.**

**Across mixed element types**, 12 screens, the aggregate looks much weaker —
0.352 → 0.433 — and the reason is a measurement artefact worth explaining rather
than hiding.

### The ground-truth convention finding

An early run showed **100% click accuracy alongside 0% IoU@50**, which is not a
combination a model failure produces. Investigation: for tabs, menu-bar items and
sidebar rows, our ground-truth box is the full *clickable region* (a 180px-wide
sidebar row), while the model boxes the visible *text* ("Archive"). Both are
defensible answers to "where is it". IoU there measures our labelling convention,
not the model.

So `Element.box_kind` now splits targets into `tight` (buttons, fields — IoU is
meaningful) and `region` (nav rows, menus, tabs — score on click accuracy
instead), and the benchmark reports both. In the 12-screen run, every `tight`
target improved sharply under refine (Refresh 0.34→0.95, Tags field 0.64→0.96,
Settings 0.62→0.83) while every `region` target stayed flat (Trash 0.09→0.08,
Spam 0.11→0.10). The aggregate hid a clean effect.

See `docs/METHOD.md`.

### Click accuracy

**100% across all 12 mixed-target trials and all 6 button trials**, both variants.
Every predicted centre landed inside the true element. On a 24-screen coarse run
there was one miss out of 24 (a "Window" menu item).

Click accuracy is what matters for acting on a screen; IoU is what matters for
cropping, redaction and highlighting. We report both because a loose box with a
correct centre is a success for one purpose and a failure for the other.

### Knowing when to refuse

Asked for `"the Print button"` on a screen with no Print button, Glance returns
nothing rather than pointing at the nearest plausible target. For an automation
tool this matters as much as accuracy: a miss stops the workflow, a false positive
clicks the wrong thing.

`scripts/distractors.py` measures this systematically — see `runs/distractors.json`.

---

## Hardware and observed resources

Everything below was measured on the machine that produced the results.

| | |
|---|---|
| Machine | MacBook Pro, Apple M1, 8 GB unified memory |
| OS | macOS (Darwin 24.5.0), arm64 |
| Cores | 8 physical / 8 logical |
| Python | 3.14.7 |
| torch / transformers | 2.14.0 / 5.17.0 |
| Accelerator | MPS (unified memory, shared with system RAM) |

**Weights resident:** ~2.1 GB (4-bit MLX). The bf16 checkpoint is ~7.5 GB.

**Latency**, steady state after discarding one warm-up call:

| Workload | View size | Prompt tokens | Time |
|---|---|---|---|
| Small image, short answer | 644x392 | 347 | 2.8s |
| Full screenshot, short answer | 1260x784 | 1285 | 8.4s |
| Verification crop | 840x252 | 302 | 2.6s |
| `find` (coarse only) | — | — | ~9.8s |
| `find` (coarse + refine + verify) | — | — | ~20s |

First call of a session costs an extra 1.5–5s (MLX kernel compilation, weights
paging in). The benchmark discards one warm-up call before timing anything;
averaging it in would silently inflate every variant.

**Cost is driven by input pixels, not output length.** 347 prompt tokens → 2.8s,
1285 → 8.4s, both generating 16 tokens. Consequence: the verification pass uses
the *smallest* pixel budget, since there is no fine detail to recover in a 96px
crop of one button. Measured: 2.55s at `min_pixels` vs 8.34s at full screen, and
doubling the budget added 1.2s for no accuracy gain.

**Resolution above ~1 megapixel is free.** 1280x800, 1920x1200 and 2560x1600 all
clamp to the same 1260x784 view and the same 1285 prompt tokens, so a 4K
screenshot costs what a 1080p one does.

### The binding constraint was memory, not parameters

We are at **62.6%** of the parameter cap and could not run the model in bf16 at
all: 7.5 GB of weights does not fit in 8 GB of unified memory alongside macOS.
The 4-bit MLX build (~2.1 GB) is what made this project runnable.

Quantisation changes bytes per weight, not the number of weights, so the 3.75B
total stands and `glance budget` audits the original HF repo regardless of which
build executes. Stated plainly: **the 6B cap was never what limited this project
on this hardware.**

At ~20s per grounded click, Glance is a batch and analysis tool here, not
something to put in a live interaction loop. On a 16 GB+ machine the bf16 HF
backend should be both faster and more accurate.

---

## Install and run

Requires Python 3.10+ and about 3 GB of disk for weights.

    git clone https://github.com/kru2710shna/-Human-Computer-Lab-.git
    cd -Human-Computer-Lab-
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt && pip install -e .

Apple silicon (recommended, and required under 16 GB RAM):

    pip install mlx-vlm

Then:

    glance hardware                    # what this machine is
    glance budget                      # prove the parameter constraint
    glance gen-data --n 3 --out-dir assets/generated
    glance find assets/generated/ui_000.png "the blue Share button" --annotate out/find.png
    python -m pytest tests -q          # 89 tests, no model needed

Useful flags: `--no-refine` (about 2x faster, looser boxes), `--no-verify` (skips
confidence), `--backend hf` (forces transformers), `--threshold 0.7`.

`glance find screen "..."` captures your display instead of reading a file
(needs `pip install mss` and Screen Recording permission).

---

## Components we did not write, and what we own about them

**Qwen2.5-VL-3B-Instruct** (Alibaba) is the only learned component. We chose it,
audited it, and own its behaviour in this system: the grounding quality, the
false-positive profile, and the failure modes below are ours to report. Check the
model card's licence before commercial use.

**mlx-community/Qwen2.5-VL-3B-Instruct-4bit** is a community quantisation we did
not produce or verify against the original. All reported accuracy is from this
build; a bf16 comparison would likely differ and we could not run one (see below).

**mlx-vlm** provides Apple-silicon inference. It emits a
`Kwargs passed to processor.__call__` warning on every call under transformers
5.x, indicating some processor kwargs may be silently dropped. Harmless in our
testing but a real version-coupling risk.

**The benchmark dataset is ours**, generated by `glance/bench/dataset.py`. We
render the screens, so every box is known to the pixel — no annotation noise, no
licensing question. The tradeoff: synthetic UIs are cleaner than real ones, so
**absolute accuracy here is optimistic**. What it measures reliably is the
relative comparison between pipeline variants on identical inputs.

This README, the code, and the tests were written with AI assistance. Every
number in it was produced by running the code on the hardware described.

---

## Completed

- Parameter audit from checkpoint headers, with component breakdown, tied-weight
  handling, negative control, and non-zero exit on violation
- Affine coordinate ledger with independent tests
- Zoom-refine grounding, measured: +70% IoU on tight-boxed elements
- Logit-level verification with abstention (HF backend)
- Six tasks plus free-form VQA
- Two backends (MLX, transformers) behind one interface
- Synthetic benchmark with exact ground truth; accuracy, latency, memory
- Self-contained HTML reports and annotated PNGs
- 89 tests, all passing, none requiring a model
- Hardware and resource reporting with warm-up separation
- False-positive measurement: 0/8 false positives on absent targets

---

## Layout

    glance/
      config.py      model registry, the 6B constant
      budget.py      parameter audit
      imaging.py     coordinate ledger, smart resize, crops
      parsing.py     defensive parsing of VLM output
      pipeline.py    zoom-refine + verification engine
      tasks.py       the six tasks
      render.py      annotated PNGs, HTML reports
      telemetry.py   hardware and resource measurement
      cli.py         the glance command
      backends/      base / hf / mlx_backend
      bench/         dataset, metrics, runner
    tests/           89 tests, no model required
    scripts/         distractors.py, summarise.py
    docs/METHOD.md   the ground-truth convention finding