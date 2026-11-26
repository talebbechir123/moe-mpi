"""MoE vs dense comparison figures and summary table.

  python compare.py --moe results/cmp_moe --dense results/cmp_dense

Reads <name>.json (metrics) and <name>.out (parameter counts) for each run and writes:
  results/fig_moe_vs_dense.png   validation loss vs step, vs time, and the gap
  results/fig_router.png         MoE routing health (dropped tokens, expert imbalance)
  results/summary.md             table for the README
"""
import argparse, json, math, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOE, DENSE = "#257de0", "#f47230"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.titlecolor": INK,
})


def load(prefix):
    d = json.load(open(prefix + ".json"))
    a, ranks = d["args"], d["ranks"]
    head = open(prefix + ".out", errors="ignore").read()
    m = re.search(r"shared=([\d,]+) experts=([\d,]+)", head)
    shared = int(m.group(1).replace(",", ""))
    experts_per_rank = int(m.group(2).replace(",", ""))
    mlp_one = 8 * a["d"] * a["d"]
    if a["mode"] == "dense":
        total = active = shared  # dense MLPs are in "shared"
    else:
        total = shared + experts_per_rank * ranks
        active = shared + a["layers"] * mlp_one  # top-1
    tok_per_step = a["batch"] * a["ctx"] * ranks
    tok_s = float(np.median([r["tok_s"] for r in d["log"] if r["step"] > 1]))
    val = d.get("val") or []
    return dict(a=a, log=d["log"], val=val, total=total, active=active,
                tok_per_step=tok_per_step, tok_s=tok_s)


def smooth(y, k=15):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


ap = argparse.ArgumentParser()
ap.add_argument("--moe", default="results/cmp_moe")
ap.add_argument("--dense", default="results/cmp_dense")
args = ap.parse_args()
M, D = load(args.moe), load(args.dense)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
fig.subplots_adjust(left=0.05, right=0.97, top=0.80, bottom=0.14, wspace=0.28)
fig.text(0.05, 0.955, "Mixture-of-Experts vs dense model at equal compute per token",
         fontsize=14, weight="bold", color=INK)
fig.text(0.05, 0.895, f"Same data, same batches, same steps. MoE: {M['total']/1e6:.1f}M parameters, "
         f"{M['active']/1e6:.1f}M active per token.  Dense: {D['total']/1e6:.1f}M parameters.  "
         f"Validation loss on held-out TinyStories (lower is better).", fontsize=9.5, color=INK2)

def val_xy(R, key):
    s = np.array([v["step"] for v in R["val"]], float)
    y = np.array([v["val_loss"] for v in R["val"]])
    if key == "time":   # median throughput
        s = s * R["tok_per_step"] / R["tok_s"] / 60
    return s, y

for i, (key, title, xlabel) in enumerate([
        ("step", "Validation loss vs training step", "training step"),
        ("time", "Validation loss vs training time", "minutes (at steady-state speed)")]):
    a_ = ax[i]
    for R, c, name in [(D, DENSE, "Dense"), (M, MOE, "MoE")]:
        x, y = val_xy(R, key)
        a_.plot(x, y, color=c, marker="o", ms=5, mec="white", mew=1.5, label=name)
        dy = 9 if name == "Dense" else -9
        a_.annotate(f"{name}  {y[-1]:.3f}", (x[-1], y[-1]), xytext=(8, dy),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)
    lo = min(M["val"][-1]["val_loss"], D["val"][-1]["val_loss"])
    a_.set_ylim(lo - 0.08, min(3.0, max(v["val_loss"] for v in M["val"] + D["val"]) + 0.05))
    a_.set_title(title); a_.set_xlabel(xlabel); a_.set_ylabel("validation loss")
    a_.margins(x=0.02); a_.set_xlim(right=a_.get_xlim()[1] * 1.18)
    a_.legend(loc="upper right")

dv = {v["step"]: v["val_loss"] for v in D["val"]}
common = [(v["step"], dv[v["step"]] - v["val_loss"]) for v in M["val"] if v["step"] in dv]
st, gap = zip(*common)
a_ = ax[2]
a_.bar(st, gap, width=(st[1] - st[0]) * 0.6 if len(st) > 1 else 300, color=MOE, edgecolor="white", lw=2)
for s, g in [(st[0], gap[0]), (st[-1], gap[-1])]:
    a_.annotate(f"{g:+.3f}", (s, g), xytext=(0, 4 if g >= 0 else -12), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
a_.axhline(0, color="#9e9e9e", lw=1)
a_.set_title("MoE advantage (dense − MoE)"); a_.set_xlabel("training step")
a_.set_ylabel("validation loss difference")
a_.set_ylim(min(0, min(gap)) - 0.01, max(gap) * 1.25 + 0.005)
fig.savefig("results/fig_moe_vs_dense.png", dpi=170)

log = M["log"]
s = np.array([r["step"] for r in log])
routed = M["tok_per_step"] * M["a"]["layers"]
drop = np.array([100 * r["dropped"] / routed for r in log])
imb = np.array([r["imbalance"] for r in log])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.16, wspace=0.22)
fig.text(0.06, 0.94, "MoE routing health during training", fontsize=13, weight="bold", color=INK)
fig.text(0.06, 0.875, "The router first collapses onto a few experts during warmup, then the "
         "load-balancing loss spreads tokens evenly.", fontsize=9.5, color=INK2)
for a_, y, title, ylab, ref in [
        (ax[0], drop, "Tokens dropped (over expert capacity)", "% of routed tokens", 0),
        (ax[1], imb, "Load imbalance (busiest expert / average)", "ratio", 1)]:
    a_.plot(s, y, color=MOE, alpha=0.25, lw=1)
    a_.plot(s, smooth(y), color=MOE, lw=2)
    a_.axhline(ref, color="#9e9e9e", lw=1, ls="--")
    a_.set_title(title); a_.set_xlabel("training step"); a_.set_ylabel(ylab)
    fmt = (lambda v: f"{v:.1f}%") if ref == 0 else (lambda v: f"{v:.2f}")
    a_.annotate(f"end: {fmt(smooth(y)[-1])}", (s[-1], smooth(y)[-1]),
                xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=9, color=INK)
    pk = int(np.argmax(y))
    a_.annotate(f"peak {fmt(y[pk])} (step {s[pk]})", (s[pk], y[pk]),
                xytext=(14, -4), textcoords="offset points", fontsize=9, color=INK)
ax[1].text(s[-1], 1.0, "perfect balance", ha="right", va="bottom", fontsize=8, color=INK2)
fig.savefig("results/fig_router.png", dpi=170)

def row(name, R):
    vl = R["val"][-1]["val_loss"]
    return (f"| {name} | {R['total']/1e6:.1f}M | {R['active']/1e6:.1f}M | {vl:.3f} | "
            f"{math.exp(vl):.2f} | {R['tok_s']:,.0f} |")
mv, dvl = M["val"][-1]["val_loss"], D["val"][-1]["val_loss"]
table = "\n".join([
    "| Model | Total params | Active params / token | Val loss | Perplexity | Tokens/s |",
    "|---|---|---|---|---|---|",
    row("Dense", D), row("MoE (8 experts, top-1)", M), "",
    f"MoE perplexity is {100 * (1 - math.exp(mv) / math.exp(dvl)):.1f}% lower; "
    f"throughput is {100 * (1 - M['tok_s'] / D['tok_s']):.0f}% lower (routing + all-to-all).",
])
open("results/summary.md", "w").write(table + "\n")
print(table)
print("saved results/fig_moe_vs_dense.png, results/fig_router.png, results/summary.md")
