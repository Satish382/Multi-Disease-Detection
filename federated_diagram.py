"""
federated_diagram.py  (v5 — large text, clean spacing)
------------------------------------------------------
IEEE single-column layout with large readable fonts.
Arrow labels placed with proper clearance from boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 14))
ax.set_xlim(0, 9)
ax.set_ylim(-0.8, 14)
ax.axis("off")

TH  = 0.65
LW  = 2.0
TFS = 13
BFS = 11
LBL = 9
CAP = 11
ARROW = dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18)


def draw_block(ax, x, y, w, h, title, lines):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0",
        lw=LW, ec="black", fc="white", zorder=2))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y+h-TH), w, TH, boxstyle="square,pad=0",
        lw=LW, ec="black", fc="#d0d0d0", zorder=3))
    ax.text(x+w/2, y+h-TH/2, title,
            ha="center", va="center", fontsize=TFS,
            fontweight="bold", zorder=4)
    slot = (h - TH) / (len(lines) + 1)
    for i, ln in enumerate(lines, 1):
        ax.text(x+w/2, y+h-TH - i*slot, ln,
                ha="center", va="center", fontsize=BFS, zorder=4)


def arr(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(**ARROW), zorder=5)


# ── Hospitals (left, stacked) ─────────────────────────────────────────────────
HW, HH = 3.0, 2.5
HX     = 0.15

hosp_data = [
    ("Hospital A", 10.8),
    ("Hospital B",  7.6),
    ("Hospital C",  4.4),
    ("Hospital D",  1.2),
]
body = [
    "Local MRI Dataset",
    "Local CNN Model",
    "(Tumor + Alzheimer)",
    "Local Training (E Local Epochs)",
]

hcy = {}
for (label, by) in hosp_data:
    draw_block(ax, HX, by, HW, HH, label, body)
    hcy[label] = by + HH / 2

# ── Central server (right, full height) ───────────────────────────────────────
SW, SH = 4.2, 12.8
SX     = 4.65
SY     = 0.75

server_lines = [
    "Global Model",
    "",
    "Federated Averaging",
    "(FedAvg)",
    "",
    "",       # equation zone
    "",
    "",
    "Communication Rounds",
    "(t = 1, 2, ..., T)",
]
draw_block(ax, SX, SY, SW, SH, "Central Server", server_lines)

# ── Equation (large, boxed) ───────────────────────────────────────────────────
eq_y = SY + SH * 0.43
ax.text(SX + SW/2, eq_y,
        r"$w^{(t+1)} = \sum_{k} \frac{n_k}{n}\, w_k^{(t)}$",
        ha="center", va="center", fontsize=16, zorder=6,
        bbox=dict(boxstyle="round,pad=0.55", fc="white", ec="black", lw=1.5))

# ── Arrows with clean label placement ─────────────────────────────────────────
HOSP_R = HX + HW        # right edge of hospitals
SRV_L  = SX             # left edge of server
MID_X  = (HOSP_R + SRV_L) / 2

labels_order = ["Hospital A", "Hospital B", "Hospital C", "Hospital D"]

for idx, label in enumerate(labels_order):
    cy = hcy[label]

    fwd_lbl = ("Local Model Weights\n(No Raw Data Transfer)"
               if idx == 0 else "Local Weights")

    # Forward arrow (hospital → server) — upper
    yf = cy + 0.35
    arr(ax, HOSP_R + 0.05, yf, SRV_L - 0.05, yf)
    ax.text(MID_X, yf + 0.15, fwd_lbl,
            ha="center", va="bottom", fontsize=LBL, style="italic", zorder=5)

    # Return arrow (server → hospital) — lower
    yb = cy - 0.35
    arr(ax, SRV_L - 0.05, yb, HOSP_R + 0.05, yb)
    ax.text(MID_X, yb - 0.15, "Updated Global Model",
            ha="center", va="top", fontsize=LBL, style="italic", zorder=5)

# ── Caption ───────────────────────────────────────────────────────────────────
ax.text(4.5, -0.35,
        "Fig. 2. Federated learning architecture for multi-disease MRI classification.",
        ha="center", va="top", fontsize=CAP)

plt.tight_layout(pad=0.15)
plt.savefig("federated_diagram.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved -> federated_diagram.png")
