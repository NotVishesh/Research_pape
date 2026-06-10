"""
Generate comparison chart from results/results.csv for the paper.
Run after baseline.py and optimized.py have both completed.
"""
import csv
import os
import sys

os.makedirs("results", exist_ok=True)
CSV_PATH = "results/results.csv"

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run baseline.py and optimized.py first.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

rows = []
with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("No data in results.csv yet.")
    sys.exit(1)

DISPLAY_NAMES = {
    "baseline": "Baseline\n(no opt.)",
    "attn_slicing": "Attention\nSlicing",
    "cpu_offload": "CPU\nOffload",
}

names  = [DISPLAY_NAMES.get(r["experiment"], r["experiment"]) for r in rows]
times  = [float(r["gen_time_s"]) for r in rows]
vrams  = [float(r["peak_vram_gb"]) for r in rows]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(names)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

bars1 = ax1.bar(names, vrams, color=colors, edgecolor="white", linewidth=0.5)
ax1.set_title("Peak VRAM Usage", fontweight="bold")
ax1.set_ylabel("GB")
ax1.set_ylim(0, max(vrams) * 1.25)
for bar, val in zip(bars1, vrams):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(vrams) * 0.02,
        f"{val:.2f} GB",
        ha="center", va="bottom", fontsize=9,
    )

bars2 = ax2.bar(names, times, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_title("Generation Time", fontweight="bold")
ax2.set_ylabel("Seconds")
ax2.set_ylim(0, max(times) * 1.25)
for bar, val in zip(bars2, times):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(times) * 0.02,
        f"{val:.1f}s",
        ha="center", va="bottom", fontsize=9,
    )

fig.suptitle(
    "EMOD: Memory Optimization Comparison\nStable Diffusion 1.5 · 512×512 · RTX 4050 Laptop GPU (6 GB)",
    fontsize=11,
)
plt.tight_layout()
out = "results/comparison_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out}")

print("\n--- Summary Table ---")
print(f"{'Experiment':<20} {'Time (s)':>10} {'VRAM (GB)':>12}")
print("-" * 44)
for r in rows:
    print(f"{r['experiment']:<20} {float(r['gen_time_s']):>10.2f} {float(r['peak_vram_gb']):>12.3f}")
