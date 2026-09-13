"""Turn the newest benchmark JSON into the markdown tables used in the README."""
import glob, json, sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("runs/bench-*.json"))[-1]
d = json.load(open(path))
print(f"Source: {path}\n")

print("| Variant | Click acc | IoU mean (all) | IoU mean (tight) | IoU>=0.75 (tight) | Median s |")
print("|---|---|---|---|---|---|")
for name, s in d["summary"].items():
    a = s["overall"]
    t_mean = a.get("tight_iou_mean", float("nan"))
    t_75 = a.get("tight_iou_at_75", float("nan"))
    print(f"| {name} | {a['click_accuracy']:.0%} | {a['iou_mean']:.3f} | "
          f"{t_mean:.3f} | {t_75:.0%} | {a['latency_s_median']:.1f} |")

print("\n| Variant | box_kind | n | IoU mean | Click acc |")
print("|---|---|---|---|---|")
for name, s in d["summary"].items():
    for kind, g in s.get("by_box_kind", {}).items():
        print(f"| {name} | {kind} | {g['n']} | {g['iou_mean']:.3f} | {g['click_accuracy']:.0%} |")

print("\n| Variant | Role | n | IoU mean |")
print("|---|---|---|---|")
for name, s in d["summary"].items():
    for role, g in s.get("by_role", {}).items():
        print(f"| {name} | {role} | {g['n']} | {g['iou_mean']:.3f} |")