"""Command line interface.

Kept on argparse rather than a framework: one less dependency in a project whose
whole claim is that it runs locally with a known, auditable footprint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import imaging, render, tasks
from .backends import available_backends, build_backend
from .budget import audit, audit_pipeline, verify_runtime
from .config import DEFAULT_MODEL, MODELS, PARAM_BUDGET, get_spec
from .pipeline import Engine, GroundConfig
from .telemetry import dump, hardware_report, measure


def _load(path: str):
    if path in ("-", "screen"):
        return imaging.capture_screen()
    p = Path(path)
    if not p.exists():
        sys.exit(f"No such file: {p}")
    return imaging.load_image(p)


def _engine(args) -> Engine:
    be = build_backend(args.model, backend=args.backend, device=args.device,
                       dtype=args.dtype)
    cfg = GroundConfig(verify=not args.no_verify,
                       accept_threshold=args.threshold)
    if args.no_refine:
        cfg.zoom_skip_area_frac = 0.0  # effectively disables the zoom pass
    return Engine(be, cfg)


def _emit(args, result: dict, image=None, title: str = "Glance") -> None:
    clean = tasks.strip_images(result)
    if args.json:
        print(json.dumps(clean, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(clean, indent=2, default=str))
        print(f"wrote {args.out}", file=sys.stderr)
    if args.annotate and image is not None:
        items = render._items_from(result)
        p = render.save_annotated(image, items, args.annotate)
        print(f"wrote {p}", file=sys.stderr)
    if args.report and image is not None:
        p = render.report_html(args.report, title, image, clean,
                               meta={"model": args.model, "backend": args.backend})
        print(f"wrote {p}", file=sys.stderr)


# ------------------------------------------------------------------ commands
def cmd_budget(args):
    if args.all:
        payload = audit_pipeline([get_spec(k) for k in MODELS if k in args.all])
    else:
        rep = audit(get_spec(args.model) if args.model in MODELS else args.model)
        payload = rep.to_dict()
        if args.verify_runtime:
            be = build_backend(args.model, backend=args.backend)
            n = be.runtime_param_count()
            if n is not None:
                payload["runtime_check"] = verify_runtime(rep, n)
    print(json.dumps(payload, indent=2))
    ok = payload.get("within_budget", True)
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{payload.get('total_parameters', payload.get('path_total_parameters', 0)):,} / "
          f"{PARAM_BUDGET:,} parameters", file=sys.stderr)
    return 0 if ok else 1


def cmd_hardware(args):
    print(json.dumps(hardware_report(), indent=2))
    print(json.dumps({"backends_available": available_backends()}, indent=2))
    return 0


def cmd_describe(args):
    img = _load(args.image)
    eng = _engine(args)
    with measure("describe") as m:
        res = tasks.describe(eng, img)
    res["resources"] = m.to_dict()
    if not args.json and not args.out:
        r = res["result"]
        print(f"app     : {r.get('app')}")
        print(f"screen  : {r.get('screen')}")
        print(f"summary : {r.get('summary')}")
        for s in r.get("state", []):
            print(f"  state : {s}")
        for a in r.get("actions", []):
            print(f"  action: {a}")
    _emit(args, res, img, "Screen description")
    return 0


def cmd_ocr(args):
    img = _load(args.image)
    eng = _engine(args)
    with measure("ocr") as m:
        res = tasks.ocr(eng, img, tiles=args.tiles)
    res["resources"] = m.to_dict()
    if not args.json and not args.out:
        print(res["text"])
    _emit(args, res, img, "Text extraction")
    return 0


def cmd_find(args):
    img = _load(args.image)
    eng = _engine(args)
    with measure("find") as m:
        res = tasks.find(eng, img, args.target, all_matches=args.all,
                         fallback_tiles=not args.no_tiles,
                         verify=not args.no_verify)
    res["resources"] = m.to_dict()
    if not args.json and not args.out:
        if not res["hits"]:
            print(f"not found: {args.target}")
        for h in res["hits"]:
            c = "" if h["confidence"] is None else f"  confidence {h['confidence']:.2f}"
            print(f"click ({h['click'][0]:.0f}, {h['click'][1]:.0f})  "
                  f"box {h['box']}  [{h['stage']}]{c}")
    _emit(args, res, img, f"Find: {args.target}")
    return 0 if res.get("hits") else 1


def cmd_guide(args):
    img = _load(args.image)
    eng = _engine(args)
    with measure("guide") as m:
        res = tasks.guide(eng, img, args.question, max_steps=args.max_steps)
    res["resources"] = m.to_dict()
    if not args.json and not args.out:
        for s in res["steps"]:
            mark = {True: "o", False: "?", None: " "}[s.get("resolved")]
            line = f" {mark} {s['n']}. {s['instruction']}"
            if s.get("hit"):
                c = s["hit"]["click"]
                line += f"   -> click ({c[0]:.0f}, {c[1]:.0f})"
            elif s.get("target"):
                line += f"   -> '{s['target']}' not located on this screen"
            print(line)
        print(f"\n{res['resolved']}/{res['groundable']} steps grounded to visible elements",
              file=sys.stderr)
    _emit(args, res, img, f"Guide: {args.question}")
    return 0


def cmd_diff(args):
    before, after = _load(args.before), _load(args.after)
    eng = _engine(args)
    with measure("diff") as m:
        res = tasks.diff(eng, before, after)
    res["resources"] = m.to_dict()
    canvas = res.get("_canvas")
    if not args.json and not args.out:
        print(res.get("summary") or "(no summary)")
        for c in res["changes"]:
            print(f"  [{c['side']:5s}] {c['description']}")
    _emit(args, res, canvas, "Screen diff")
    return 0


def cmd_redact(args):
    img = _load(args.image)
    eng = _engine(args)
    with measure("redact") as m:
        res = tasks.redact(eng, img, kinds=args.kinds, method=args.method,
                           verify=args.verify_pii)
    res["resources"] = m.to_dict()
    out = args.save or "redacted.png"
    res["_image"].save(out)
    if not args.json:
        for f in res["findings"]:
            kept = "masked " if f.get("kept", True) else "skipped"
            print(f"  {kept} {f['kind']:10s} {f['box']}")
        print(f"\n{res['redacted_count']} region(s) masked -> {out}", file=sys.stderr)
        print("This is a model reading pixels, not a validated PII detector. "
              "Review before sharing.", file=sys.stderr)
    _emit(args, res, img, "Redaction")
    return 0


def cmd_ask(args):
    img = _load(args.image)
    eng = _engine(args)
    res = tasks.ask(eng, img, args.question)
    if not args.json and not args.out:
        print(res["answer"])
    _emit(args, res, img, "Question")
    return 0


def cmd_gen_data(args):
    from .bench import dataset

    recs = dataset.build(n=args.n, out=args.out_dir, size=tuple(args.size))
    meta = [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]
    dump(Path(args.out_dir) / "ground_truth.json", {"screens": meta})
    print(f"{len(recs)} screens -> {args.out_dir}")
    return 0


def cmd_bench(args):
    from .bench.run import run

    run(n=args.n, model=args.model, backend=args.backend, out=args.out_dir,
        variants=args.variants, targets_per_screen=args.targets, seed=args.seed)
    return 0


def cmd_ui(args):
    from .app import launch

    launch(model=args.model, backend=args.backend, share=args.share)
    return 0

def build_parser() -> argparse.ArgumentParser:
    # Options live on a parent parser so they work BEFORE or AFTER the
    # subcommand. Without this, argparse only accepts them before, which is a
    # constant source of "unrecognized arguments" for anyone typing naturally.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    common.add_argument("--backend", default="auto", choices=["auto", "hf", "mlx"])
    common.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    common.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    common.add_argument("--json", action="store_true", help="print full JSON result")
    common.add_argument("--out", help="write JSON result to this path")
    common.add_argument("--annotate", help="write an annotated PNG to this path")
    common.add_argument("--report", help="write a self-contained HTML report to this path")
    common.add_argument("--no-verify", action="store_true",
                        help="skip the verification pass (faster, no confidence score)")
    common.add_argument("--no-refine", action="store_true",
                        help="skip the zoom-refine pass (about 2x faster, looser boxes)")
    common.add_argument("--threshold", type=float, default=0.55,
                        help="confidence below which a hit is flagged as unreliable")

    p = argparse.ArgumentParser(
        prog="glance", parents=[common],
        description="Local screenshot understanding and UI grounding, "
                    f"under a {PARAM_BUDGET/1e9:.0f}B total-parameter budget.")
    sub = p.add_subparsers(dest="cmd", required=True)
    add = lambda name, help_: sub.add_parser(name, help=help_, parents=[common])

    s = add("budget", "audit total parameters against the cap")
    s.add_argument("--all", nargs="*", help="audit several models as one pipeline")
    s.add_argument("--verify-runtime", action="store_true",
                   help="load the model and cross-check the header count")
    s.set_defaults(fn=cmd_budget)

    s = add("hardware", "report hardware and available backends")
    s.set_defaults(fn=cmd_hardware)

    s = add("describe", "structured read of a screen")
    s.add_argument("image", help="path to an image, or 'screen' to capture")
    s.set_defaults(fn=cmd_describe)

    s = add("ocr", "extract text with boxes, in reading order")
    s.add_argument("image")
    s.add_argument("--tiles", type=int, default=1,
                   help="NxN tiling for small text (costs N^2 model calls)")
    s.set_defaults(fn=cmd_ocr)

    s = add("find", "locate an element and return a click point")
    s.add_argument("image")
    s.add_argument("target", help='e.g. "the blue Save button"')
    s.add_argument("--all", action="store_true", help="find every match")
    s.add_argument("--no-tiles", action="store_true",
                   help="skip the tile-sweep fallback when the first pass is weak")
    s.set_defaults(fn=cmd_find)

    s = add("guide", "plan steps and ground each one to the screen")
    s.add_argument("image")
    s.add_argument("question")
    s.add_argument("--max-steps", type=int, default=6)
    s.set_defaults(fn=cmd_guide)

    s = add("diff", "what changed between two screens")
    s.add_argument("before")
    s.add_argument("after")
    s.set_defaults(fn=cmd_diff)

    s = add("redact", "find and mask sensitive values")
    s.add_argument("image")
    s.add_argument("--kinds", nargs="*", default=None)
    s.add_argument("--method", default="blur", choices=["blur", "pixelate", "solid"])
    s.add_argument("--save", help="output PNG path (default: redacted.png)")
    s.add_argument("--verify-pii", action="store_true",
                   help="verify each finding before masking (slower, fewer false hits)")
    s.set_defaults(fn=cmd_redact)

    s = add("ask", "free-form question about a screen")
    s.add_argument("image")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    s = add("gen-data", "generate synthetic UI screens with ground truth")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--out-dir", default="assets/generated",
                   help="directory for generated screens")
    s.add_argument("--size", nargs=2, type=int, default=[1280, 800])
    s.set_defaults(fn=cmd_gen_data)

    s = add("bench", "run the accuracy and latency benchmark")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--out-dir", default="runs", help="directory for benchmark output")
    s.add_argument("--targets", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--variants", nargs="*",
                   default=["coarse", "coarse+verify", "refine", "full"])
    s.set_defaults(fn=cmd_bench)

    s = add("ui", "launch the local Gradio interface")
    s.add_argument("--share", action="store_true")
    s.set_defaults(fn=cmd_ui)

    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())