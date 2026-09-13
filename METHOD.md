# Method notes

## Why the coordinate ledger exists

Zoom-refine crops from the original screenshot and upscales. Without exact
bookkeeping, a box found in that crop cannot be mapped back to screen pixels, and
the whole approach collapses.

Every image carries `info["glance_tf"] = (ox, oy, sx, sy)`, mapping its own pixels
to the original: `orig_x = ox + x * sx`. `crop` adds the offset, `prepare` scales
the factors, and both compose, so arbitrarily nested views map home in one step.

This is tested without a model (`tests/test_imaging.py`): nested crops, the full
crop→upscale→map-back path, and the large-screen clamp. If this were wrong, every
IoU number in the benchmark would be wrong in a way no model test would catch.

## Feeding the model exactly the pixels we think we are

`fit_dims` reimplements the Qwen "smart resize" rule (snap to a multiple of the
patch factor, clamp between min and max pixels). Feeding an image that already
satisfies it makes the processor's own resize a no-op, so the pixels the model saw
are the pixels we measured against — on every backend.

A side effect, confirmed by measurement: 1280x800, 1920x1200 and 2560x1600 all
clamp to 1260x784 and 1285 prompt tokens. Screen resolution above ~1 MP is free.

## The ground-truth convention finding

The first benchmark returned **100% click accuracy and 0% IoU@50** — not a
combination model failure produces. A model that is wrong about *where* something
is does not land every centre inside the true box.

The cause was in our labels. `dataset.py` defines a sidebar item's box as the full
180px-wide clickable row and a menu item's as the full menu-bar height. The model
boxes the visible text. Both answer "where is it" correctly; they answer different
questions. IoU between them is near zero by construction.

Rather than quietly relabel or drop these, `Element.box_kind` makes the split
explicit:

- **tight** (button, field) — the drawn rectangle *is* the element. IoU meaningful.
- **region** (nav_item, menu, tab) — hit-area much larger than visible text. Score
  on click accuracy and containment.

The benchmark reports `tight_iou_*` alongside the aggregate. This is why the
headline result is stated on buttons and fields (0.536 → 0.911) rather than on the
mixed aggregate (0.352 → 0.433): the mixed number averages a real effect together
with a labelling artefact.

The general lesson: when a metric disagrees with another metric on the same
trials, suspect the metric before the model.

## Why verification reads logits

Asking a model "is this correct?" and parsing its answer measures its willingness
to agree, not its belief. Reading the next-token distribution and renormalising
over the Yes/No token groups gives a number that can be thresholded and that
supports abstention (`None` when the model wants to say something else entirely).

`confidence_analysis` in `bench/metrics.py` reports precision against coverage
across thresholds, which is the honest presentation: a system that is 70% accurate
but knows *which* 70% is more useful than one that is 75% accurate and equally
sure about everything.

Implemented in `backends/hf.py`. The MLX backend has no logit access, so on the
hardware used for these results confidence degrades to binary. Stated as a
limitation rather than presented as calibrated.

## Why the verification crop is small

Latency tracks input pixels, not output tokens (347 tokens → 2.8s, 1285 → 8.4s,
both generating 16 tokens). Verification generates a single token, so it is almost
pure prefill cost.

Measured: 2.55s at `min_pixels` against 8.34s at full-screen budget, with no
accuracy change — there is no fine detail to recover in a 96px crop of one button.
Doubling the budget added 1.2s for nothing. `find` went from ~25s to ~20s.

## Benchmark protocol

- One warm-up call discarded before timing (4.3s first call vs 2.8s steady).
- All variants see the same screens and the same targets in the same order.
- Targets sampled deterministically from the seed; reruns are comparable.
- Descriptors that are ambiguous on their screen are removed: if two things are
  both "the Save button", a miss is a bad question, not a model failure.
- Difficulty is a scale parameter (1.0 / 0.75 / 0.55) shrinking all controls, so
  accuracy can be read against target size.

## What the benchmark still does not measure

Every trial asks for something that exists, so `found_rate` is recall with no
false-positive check. A system that always emits a plausible box would score 100%
and be useless. `scripts/distractors.py` asks for absent elements and reports the
correct-rejection rate; it is a small run and should be extended.