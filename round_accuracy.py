"""
round_accuracy.py
-----------------
Plots "Global Test Accuracy vs Federated Round" for the brain-tumor
federated learning model (NUM_CLIENTS=10, NUM_ROUNDS=3).

The per-round accuracy values come directly from the notebook (Cell 18),
which logged the LOCAL training accuracy of each of the 5 selected clients
per round AFTER they trained on the global model.

Because the Kaggle training data is not available locally, we cannot
re-run the full federated loop.  Instead we reproduce the exact numbers
that were printed during the original Kaggle run and compute:

    round_acc[r] = mean of the 5 selected-client accuracies in round r

This is a standard proxy for global-model performance between rounds,
consistent with FedAvg literature.

Saves:  round_accuracy.png  (300 DPI, IEEE-column-width safe)

─────────────────────────────────────────────────────────────────────────────
HOW TO ADD LIVE EVALUATION TO THE NOTEBOOK TRAINING LOOP
─────────────────────────────────────────────────────────────────────────────
Insert the block below INSIDE the federated loop, right after
    model.set_weights(new_weights)   # aggregation done
and BEFORE the next round starts:

    # ── Evaluate global model on fixed test set after each round ──────────
    batch_size_eval = 32
    steps_eval      = int(len(test_paths) / batch_size_eval)
    round_correct   = 0
    round_total     = 0
    for x_batch, y_batch in datagen(test_paths, test_labels,
                                    batch_size=batch_size_eval, epochs=1):
        preds        = model.predict(x_batch)
        pred_classes = np.argmax(preds, axis=-1)
        round_correct += np.sum(pred_classes == y_batch)
        round_total   += len(y_batch)
        if round_total >= steps_eval * batch_size_eval:
            break
    global_acc = round_correct / round_total
    round_acc.append(global_acc)
    print(f"[Round {round_num+1}] Global Test Accuracy: {global_acc:.4f}")

Also declare   round_acc = []   before the federated loop starts.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Real values logged in notebook Cell 18 (Kaggle run) ──────────────────────
#
# 5 clients selected per round (50 % of 10).
# Values are each selected client's LOCAL training accuracy
# after fitting on the global-model weights for that round.

accuracies_per_round = {
    1: [0.9165, 0.9074, 0.9274, 0.9165, 0.9256],   # Round 1
    2: [0.9238, 0.9238, 0.9310, 0.9347, 0.9183],   # Round 2
    3: [0.9292, 0.9347, 0.9201, 0.9347, 0.9274],   # Round 3
}

losses_per_round = {
    1: [0.1965, 0.2428, 0.1971, 0.2180, 0.2305],
    2: [0.2053, 0.2000, 0.1778, 0.1778, 0.2388],
    3: [0.1994, 0.1905, 0.2367, 0.1814, 0.1913],
}

# ── Compute round-level global accuracy (FedAvg mean) ────────────────────────
NUM_ROUNDS = 3
rounds     = list(range(1, NUM_ROUNDS + 1))

round_acc  = [float(np.mean(accuracies_per_round[r])) for r in rounds]
round_loss = [float(np.mean(losses_per_round[r]))     for r in rounds]

# ── Print table ───────────────────────────────────────────────────────────────
print("=" * 45)
print(f"{'Round':>7}  {'Global Accuracy':>16}  {'Global Loss':>12}")
print("=" * 45)
for r, acc, loss in zip(rounds, round_acc, round_loss):
    print(f"{'Round ' + str(r):>7}  {acc:>16.4f}  {loss:>12.4f}")
print("=" * 45)
print(f"{'':>7}  Trend: {'↑' if round_acc[-1] > round_acc[0] else '↓'} "
      f"{abs(round_acc[-1]-round_acc[0])*100:.2f} pp over {NUM_ROUNDS} rounds")
print()

# ── Plot ──────────────────────────────────────────────────────────────────────
# Figure width 3.5 in → IEEE single-column; use 7.0 in for double-column.
fig, ax = plt.subplots(figsize=(5, 3.8))

# Accuracy line
ax.plot(
    rounds, [a * 100 for a in round_acc],
    marker="o", markersize=7,
    linewidth=2, color="#1f77b4",
    label="Avg. Client Training Accuracy",
)

# Annotate each point
for r, acc in zip(rounds, round_acc):
    ax.annotate(
        f"{acc*100:.2f}%",
        xy=(r, acc * 100),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center", fontsize=9, color="#1f77b4", fontweight="bold",
    )

# Formatting
ax.set_xlim(0.7, NUM_ROUNDS + 0.3)
y_min = min(round_acc) * 100 - 1.0
y_max = max(round_acc) * 100 + 1.5
ax.set_ylim(y_min, y_max)

ax.set_xticks(rounds)
ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.tick_params(axis="y", labelsize=9)

ax.set_xlabel("Communication Round", fontsize=10, labelpad=6)
ax.set_ylabel("Avg. Client Training Accuracy (%)", fontsize=10, labelpad=6)
ax.set_title("Average Client Training Accuracy vs Communication Round", fontsize=10, pad=10)

ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

OUTPUT_PNG = "round_accuracy.png"
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved → '{OUTPUT_PNG}'  (300 DPI)")
plt.show()
