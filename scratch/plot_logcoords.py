"""Scatter of the exact one-layer object in log coordinates (24.11.2026)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
d = np.load("/tmp/logcoords.npy", allow_pickle=True).item()
keys = ["flipflop s2", "parity_hand_reset s0", "parity_hand s0"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="#fcfcfb")
for ax, k in zip(axs, keys):
    r, logD, lab, _, (h, a, tau) = d[k]
    rc = r - r.mean(0)
    pc1 = rc @ np.linalg.svd(rc, full_matrices=False)[2][0]
    x = logD - logD.mean()
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(x[m], pc1[m], s=10, color=PAL[c % 6], alpha=0.55, linewidths=0, label=f"state {c}")
    ax.set_title(f"{k}  (head {h}, query {a}, τ={tau})", fontsize=10, color="#222")
    ax.set_xlabel("log mass of the prefix, log D_p (centred)", fontsize=9, color="#555")
    ax.set_ylabel("average read r_p, 1st principal direction", fontsize=9, color="#555")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors="#666")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, markerscale=1.8, loc="best")
fig.suptitle("One-layer models: what a prefix passes to one future read, in log coordinates; colour = true automaton state (3000 prefixes of length 16)", fontsize=10, color="#222")
fig.tight_layout()
fig.savefig("out/logcoords.png", dpi=130)
