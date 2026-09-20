"""
Charts for the Validation extensions (kuhn_exact, backtest,
multi_period_kelly, leduc_solver). Kept separate from the original
visualize.py so the core four-module charts stay untouched.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gto"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bankroll"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import run_backtest
from multi_period_kelly import simulate_session
from leduc_solver import LeducCFRSolver
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2c5f8a"
RED = "#a33a3a"
GREY = "#6b6b6b"
GREEN = "#3a7d5c"


def chart_backtest_trajectory():
    result = run_backtest(n_hands=300, pot=100, bet=100, kelly_fraction=0.25, seed=42)
    traj = result["trajectory"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(len(traj)), traj, color=BLUE, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Hand number")
    ax.set_ylabel("Bankroll (log scale)")
    ax.set_title("Backtest: simulated bankroll over 300 hands (quarter-Kelly)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "backtest_trajectory.png"), dpi=150)
    plt.close(fig)


def chart_kelly_drawdown_comparison():
    rng_naive = random.Random(7)
    rng_shrunk = random.Random(7)

    naive_dd, shrunk_dd = [], []
    for _ in range(2000):
        _, dd = simulate_session(rng_naive, 15, 0.55, 0.12, 1.0, kelly_scale=1.0, bankroll=1000)
        naive_dd.append(dd * 100)
    for _ in range(2000):
        _, dd = simulate_session(rng_shrunk, 15, 0.55, 0.12, 1.0, kelly_scale=0.5, bankroll=1000)
        shrunk_dd.append(dd * 100)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(naive_dd, bins=40, alpha=0.6, color=RED, label="Naive Kelly")
    ax.hist(shrunk_dd, bins=40, alpha=0.6, color=BLUE, label="Shrunk Kelly (half-sized)")
    ax.set_xlabel("Max drawdown within a session (%)")
    ax.set_ylabel("Number of sessions (out of 2,000)")
    ax.set_title("Drawdown risk: naive vs. correlation-adjusted Kelly")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kelly_drawdown_comparison.png"), dpi=150)
    plt.close(fig)


def chart_leduc_convergence():
    checkpoints = [1000, 5000, 20000, 50000, 100000, 200000, 500000]
    values = []
    for n in checkpoints:
        solver = LeducCFRSolver(seed=1)
        v = solver.train(n)
        values.append(v)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(checkpoints, values, marker="o", color=GREEN, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("CFR training iterations (log scale)")
    ax.set_ylabel("Estimated game value for Player 0")
    ax.set_title("Leduc Hold'em: CFR value stabilizing with more training")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "leduc_convergence.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating backtest trajectory chart...")
    chart_backtest_trajectory()
    print("Generating Kelly drawdown comparison chart...")
    chart_kelly_drawdown_comparison()
    print("Generating Leduc convergence chart (retrains at several checkpoints, ~1-2 min)...")
    chart_leduc_convergence()
    print(f"\nAll validation charts saved to: {OUT_DIR}")
    