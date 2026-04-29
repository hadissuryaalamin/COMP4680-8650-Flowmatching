"""
Part 1.1 - Data Visualization
Visualize swiss_roll, gaussians, circles at D=2 and D=32 (projected back to 2D)
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataloader import ToyDiffusionDataset

DATASETS = ["swiss_roll", "gaussians", "circles"]
COLORS   = {"swiss_roll": "steelblue", "gaussians": "tomato", "circles": "seagreen"}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Part 1.1 – Data Visualization", fontsize=14, fontweight="bold")

for col, name in enumerate(DATASETS):
    # --- Row 0: original 2D ---
    ds2 = ToyDiffusionDataset(name=name, dim=2)
    data2 = ds2.data.numpy()
    ax = axes[0, col]
    ax.scatter(data2[:, 0], data2[:, 1], s=2, alpha=0.4, color=COLORS[name])
    ax.set_title(f"{name}\n(D=2, original)")
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Row 1: D=32 projected back to 2D ---
    ds32 = ToyDiffusionDataset(name=name, dim=32)
    data32_np = ds32.data.numpy()
    data32_2d = ds32.to_2d(data32_np)
    ax = axes[1, col]
    ax.scatter(data32_2d[:, 0], data32_2d[:, 1], s=2, alpha=0.4, color=COLORS[name])
    ax.set_title(f"{name}\n(D=32 → projected to 2D)")
    ax.set_aspect("equal")
    ax.axis("off")

axes[0, 0].set_ylabel("D=2 original", fontsize=10)
axes[1, 0].set_ylabel("D=32 projected", fontsize=10)

plt.tight_layout()
out_path = Path(__file__).resolve().parent.parent / "part1_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
