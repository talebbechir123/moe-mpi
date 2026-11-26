"""Architecture figure: model stack + one MoE layer across 2 ranks."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, BLUE_L = "#1d4ed8", "#dbeafe"  # shared
ORG, ORG_L = "#c2410c", "#ffedd5"  # experts
YEL, YEL_L = "#a16207", "#fef9c3"  # MPI
GRN, GRN_L = "#15803d", "#dcfce7"
GRY, GRY_L, INK = "#6b7280", "#f3f4f6", "#111827"

fig = plt.figure(figsize=(14, 8.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 140); ax.set_ylim(0, 86); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=9, bold=False, color=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", color=color, linespacing=1.25)


def arrow(x1, y1, x2, y2, color=GRY, lw=1.3, style="-|>", ls="-"):
    ax.annotate("", (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


ax.text(2, 83, "MoE GPT with expert parallelism over MPI", fontsize=16, weight="bold", color=INK)
ax.text(2, 79.6, "Left: the model.  Right: what happens inside one MoE layer when the experts are split "
        "across 2 MPI ranks (8 experts, top-1 routing).", fontsize=10, color=GRY)

X, W = 4, 30
box(X, 4, W, 5, "Input tokens (BPE, vocab 4096)", GRY_L, GRY)
box(X, 12, W, 5.5, "Token + position embeddings", BLUE_L, BLUE)
ax.add_patch(FancyBboxPatch((X - 1.5, 21), W + 3, 38, boxstyle="round,pad=0.3,rounding_size=1.2",
                            fc="white", ec=INK, lw=1.4, ls="--"))
ax.text(X - 1, 61.2, "Transformer block  × 4 layers", ha="left", fontsize=10, weight="bold")
box(X, 23.5, W, 4.5, "LayerNorm", BLUE_L, BLUE)
box(X, 30.5, W, 6, "Causal multi-head attention\n(4 heads, d = 256)", BLUE_L, BLUE)
box(X, 40, W, 4.5, "LayerNorm", BLUE_L, BLUE)
box(X, 47, W, 7.5, "Mixture-of-Experts layer\nrouter + 8 expert MLPs\n(replaces the usual MLP)",
    ORG_L, ORG, bold=True)
box(X, 64, W, 4.5, "LayerNorm + linear head", BLUE_L, BLUE)
box(X, 71.5, W, 4.5, "Next-token probabilities", GRY_L, GRY)
for y1, y2 in [(9, 12), (17.5, 23.5), (28, 30.5), (36.5, 40), (44.5, 47), (54.5, 64), (68.5, 71.5)]:
    arrow(X + W / 2, y1, X + W / 2, y2)
for y, t in [(33.5, "+ residual"), (50.75, "+ residual")]:
    ax.text(X + W + 2.6, y, t, fontsize=7.5, color=GRY, rotation=90, va="center")
ax.plot([X + W + 5, 45], [54.5, 74], color=ORG, lw=1, ls=":")
ax.plot([X + W + 5, 45], [47, 5], color=ORG, lw=1, ls=":")

R0, R1, RW = 46, 94, 42
for rx, r in [(R0, 0), (R1, 1)]:
    ax.add_patch(FancyBboxPatch((rx, 5), RW, 69, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="#fafafa", ec=GRY, lw=1.2))
    e0 = 4 * r
    ax.text(rx + RW / 2, 71.3, f"MPI rank {r}", ha="center", fontsize=12, weight="bold")
    ax.text(rx + RW / 2, 69.2, f"owns experts {e0}–{e0 + 3} only", ha="center", fontsize=9,
            color=ORG, style="italic")
    box(rx + 2, 60.5, RW - 4, 6, f"Its own batch of tokens (16 × 128)\nshared weights: identical copy on every rank",
        BLUE_L, BLUE, fs=8.5)
    box(rx + 2, 52.5, RW - 4, 6.5,
        "Router: softmax score per expert → best expert (top-1)\ncapacity limit per expert; extra tokens dropped",
        BLUE_L, BLUE, fs=8.3)
    box(rx + 2, 44.5, RW - 4, 5, "Sort tokens by expert = group them by destination rank",
        GRY_L, GRY, fs=8.3)
    ew = (RW - 4 - 3 * 1.5) / 4
    for k in range(4):
        box(rx + 2 + k * (ew + 1.5), 25, ew, 7.5, f"Expert {e0 + k}\nMLP\nd→4d→d",
            ORG_L, ORG, fs=8)
    box(rx + 2, 7, RW - 4, 7,
        "Results back in original token order\n× router probability (gate) → added to residual",
        GRN_L, GRN, fs=8.3)
    arrow(rx + RW / 2, 60.5, rx + RW / 2, 59)
    arrow(rx + RW / 2, 52.5, rx + RW / 2, 49.5)

box(R0 + 1, 36.5, R1 + RW - R0 - 2, 5.5,
    "① MPI_Alltoall (counts)  →  ② MPI_Alltoallv DISPATCH: each token goes to the rank that owns its expert",
    YEL_L, YEL, fs=9, bold=True)
box(R0 + 1, 16.2, R1 + RW - R0 - 2, 4.6,
    "③ MPI_Alltoallv COMBINE: expert outputs go back to the token's home rank",
    YEL_L, YEL, fs=9, bold=True)
for rx in (R0, R1):
    arrow(rx + RW / 2, 44.5, rx + RW / 2, 42)
    arrow(rx + RW / 2, 36.5, rx + RW / 2, 32.5)
    arrow(rx + RW / 2, 25, rx + RW / 2, 20.8)
    arrow(rx + RW / 2, 16.2, rx + RW / 2, 14)
arrow(R0 + 30, 36.5, R1 + 8, 32.5, color=YEL, lw=1.6)
arrow(R1 + 12, 36.5, R0 + 34, 32.5, color=YEL, lw=1.6)

ax.text(46, 2.4, "Backward pass: the same two exchanges in reverse order (custom autograd function).",
        fontsize=8.5, color=GRY)
ax.text(46, 0.6, "Gradients: shared weights averaged with MPI_Allreduce; expert weights stay local "
        "(they already received gradients from every rank's tokens).", fontsize=8.5, color=GRY)
handles = [(BLUE_L, BLUE, "copied on every rank (data parallel)"),
           (ORG_L, ORG, "split across ranks (expert parallel)"),
           (YEL_L, YEL, "MPI communication")]
for i, (fc, ec, t) in enumerate(handles):
    ax.add_patch(FancyBboxPatch((96 + 0 * i, 81.2 - i * 2.6), 2.2, 1.6, boxstyle="round,pad=0.1", fc=fc, ec=ec))
    ax.text(99, 82 - i * 2.6, t, fontsize=8.5, va="center", color=INK)

fig.savefig("results/architecture.png", dpi=170)
fig.savefig("results/architecture.svg")
print("saved results/architecture.png and .svg")
