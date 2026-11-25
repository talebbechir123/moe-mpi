import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = sys.argv[1:] or ["results/main_ep.json"]
fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
for path in runs:
    d = json.load(open(path)); log, a = d["log"], d["args"]
    s = np.array([r["step"] for r in log])
    loss = np.array([r["loss"] for r in log])
    k = min(10, len(loss))
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    tok_per_step = a["batch"] * a["ctx"] * d["ranks"] * a["layers"]
    lab = path.split("/")[-1].replace(".json", "")
    ax[0, 0].plot(s, loss, alpha=.25); ax[0, 0].plot(s[k - 1:], smooth, label=lab)
    if d.get("val"):
        ax[0, 0].plot([v["step"] for v in d["val"]], [v["val_loss"] for v in d["val"]],
                      "o--", ms=4, label=f"{lab} validation")
    ax[0, 1].plot(s, [100 * r["dropped"] / tok_per_step for r in log], label=lab)
    ax[1, 0].plot(s, [r["imbalance"] for r in log], label=lab)
    ax[1, 1].plot(s, [r["a2a_pct"] for r in log], label=f"{lab} all-to-all")
    ax[1, 1].plot(s, [r["allreduce_pct"] for r in log], "--", label=f"{lab} allreduce")
for a_, t in zip(ax.flat, ["Training loss", "Tokens dropped (%)",
                          "Load imbalance (busiest / mean)", "% of time in MPI"]):
    a_.set_title(t, loc="left"); a_.set_xlabel("step"); a_.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/training.png", dpi=150)
print("saved results/training.png")
