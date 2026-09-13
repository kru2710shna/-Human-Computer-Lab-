"""Measure false positives: ask for elements that are NOT on screen."""
from glance.bench import dataset
from glance.bench.run import run_distractors
from glance.backends import build_backend
from glance.pipeline import Engine
import json, pathlib

recs = dataset.build(n=8, out=None)
eng = Engine(build_backend("qwen2.5-vl-3b", backend="mlx"))
r = run_distractors(eng, recs)
print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
pathlib.Path("runs").mkdir(exist_ok=True)
json.dump(r, open("runs/distractors.json", "w"), indent=2)