"""
Produces the figures and aggregated tables for the paper revision from
results_revision/results.csv. Outputs go to results_revision/figures/.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fig_scaling(df: pd.DataFrame, out_dir: Path):
    """Avg per-robot distance vs N (Test case 1 style), with std error bars."""
    g = df.groupby(["method", "n_robots"]).agg(
        mean_total=("total_distance", "mean"),
        std_total=("total_distance", "std"),
        mean_max=("max_distance", "mean"),
        std_max=("max_distance", "std"),
        success=("success", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, color in zip(["ppo", "astar"], ["tab:blue", "tab:orange"]):
        sub = g[g.method == method].sort_values("n_robots")
        axes[0].errorbar(sub.n_robots, sub.mean_total, yerr=sub.std_total,
                         label=method.upper(), marker="o", color=color, capsize=4)
        axes[1].errorbar(sub.n_robots, sub.mean_max, yerr=sub.std_max,
                         label=method.upper(), marker="o", color=color, capsize=4)
        axes[2].plot(sub.n_robots, sub.success, label=method.upper(),
                     marker="o", color=color)
    for ax, ylabel in zip(axes, [
        "Total distance (cells)", "Max per-robot distance (cells)",
        "Success rate",
    ]):
        ax.set_xlabel("Number of robots, N")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Scaling with fleet size: PPO vs A* baseline")
    fig.tight_layout()
    out_path = out_dir / "fig_scaling_ppo_vs_astar.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_per_config_bars(df: pd.DataFrame, out_dir: Path):
    """Bar chart by config (Test case 1 style) with error bars."""
    g = df.groupby(["method", "config"]).agg(
        mean_total=("total_distance", "mean"),
        std_total=("total_distance", "std"),
    ).reset_index()
    configs = sorted(g.config.unique())
    x = np.arange(len(configs))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (method, color) in enumerate(zip(["ppo", "astar"], ["tab:blue", "tab:orange"])):
        sub = g[g.method == method].set_index("config").reindex(configs).reset_index()
        ax.bar(x + (i - 0.5) * width, sub.mean_total, width=width,
               yerr=sub.std_total, label=method.upper(), color=color, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylabel("Total distance (cells)")
    ax.set_title("Per-configuration total distance: PPO vs A*")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "fig_per_config_bars.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def table_summary(df: pd.DataFrame, out_dir: Path):
    g = df.groupby(["method", "config"]).agg(
        success_mean=("success", "mean"),
        total_mean=("total_distance", "mean"),
        total_std=("total_distance", "std"),
        max_mean=("max_distance", "mean"),
        max_std=("max_distance", "std"),
        steps_mean=("steps", "mean"),
        n=("episode", "count"),
    ).reset_index()
    out_path = out_dir / "table_summary.csv"
    g.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print(g.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", type=str, default="results_revision/results.csv")
    p.add_argument("--out-dir", type=str, default="results_revision/figures")
    args = p.parse_args()
    df = pd.read_csv(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_scaling(df, out_dir)
    fig_per_config_bars(df, out_dir)
    table_summary(df, out_dir)


if __name__ == "__main__":
    main()
