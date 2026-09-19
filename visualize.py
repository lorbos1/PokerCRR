"""
Visualizations for PokerCRR
------------------------------
Generates four charts that summarize what the project actually shows:

1. equity_comparison.png  -- equity of the two worked examples in main.py
2. gto_strategy.png       -- Kuhn solver's learned bet/bluff frequencies
3. cfr_convergence.png    -- how the CFR solver's game-value estimate
                              converges to the known theoretical value
                              as training iterations increase
4. kelly_curve.png        -- optimal Kelly stake as a function of equity,
                              for a fixed pot-size bet

Run this after main.py's modules are importable (same folder layout).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "equity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pricing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gto"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bankroll"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monte_carlo import estimate_equity
from kuhn_solver import KuhnCFRSolver
from kelly import kelly_from_pot_odds

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# A calm, finance-report-appropriate palette (no neon defaults)
BLUE = "#2c5f8a"
RED = "#a33a3a"
GREY = "#6b6b6b"
GREEN = "#3a7d5c"


def chart_equity_comparison():
    hands = [
        ("KK on 7-2-9 board", ["Kh", "Kd"], ["7c", "2d", "9s"]),
        ("72o on A-K-Q board", ["7c", "2d"], ["Ah", "Kd", "Qs"]),
    ]
    labels, equities = [], []
    for label, hole, board in hands:
        r = estimate_equity(hole, board=board, n_opponents=1, n_sims=20000, seed=42)
        equities.append(r["win"] + r["tie"] / 2)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, equities, color=[BLUE, RED], width=0.5)
    ax.axhline(0.333, color=GREY, linestyle="--", linewidth=1,
               label="Pot odds required to call (33.3%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Equity")
    ax.set_title("Equity vs. break-even threshold (Monte Carlo, 20,000 sims)")
    for bar, val in zip(bars, equities):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}",
                 ha="center", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "equity_comparison.png"), dpi=150)
    plt.close(fig)


def chart_gto_strategy(solver):
    order = ["J", "Q", "K", "J | b", "J | p", "Q | b", "Q | p",
              "K | b", "K | p", "J | pb", "Q | pb", "K | pb"]
    card_name = {"0": "J", "1": "Q", "2": "K"}
    bet_freqs = []
    for label in order:
        card = [k for k, v in card_name.items() if v == label.split(" ")[0]][0]
        history = label.split(" | ")[1] if " | " in label else ""
        key = card + history
        if key in solver.node_map:
            avg = solver.node_map[key].get_average_strategy()
            bet_freqs.append(avg[1])
        else:
            bet_freqs.append(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [RED if f > 0.5 else BLUE for f in bet_freqs]
    ax.bar(order, bet_freqs, color=colors)
    ax.axhline(1 / 3, color=GREY, linestyle="--", linewidth=1,
               label="1/3 (known Jack opening-bluff bound)")
    ax.set_ylabel("Bet/call frequency")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kuhn Poker: CFR-learned equilibrium strategy by information set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gto_strategy.png"), dpi=150)
    plt.close(fig)


def chart_cfr_convergence():
    checkpoints = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    values = []
    for n in checkpoints:
        solver = KuhnCFRSolver(seed=1)
        v = solver.train(n)
        values.append(v)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(checkpoints, values, marker="o", color=BLUE, linewidth=1.5)
    ax.axhline(-1 / 18, color=RED, linestyle="--", linewidth=1,
               label="Theoretical value (-1/18)")
    ax.set_xscale("log")
    ax.set_xlabel("CFR training iterations (log scale)")
    ax.set_ylabel("Estimated game value for Player 1")
    ax.set_title("CFR convergence to the known Kuhn Poker equilibrium")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "cfr_convergence.png"), dpi=150)
    plt.close(fig)


def chart_kelly_curve():
    pot, bet = 100, 100  # pot-size bet
    equities = np.linspace(0.0, 1.0, 200)
    fractions = [kelly_from_pot_odds(e, pot, bet) for e in equities]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(equities, fractions, color=GREEN, linewidth=2)
    ax.axvline(0.5, color=GREY, linestyle="--", linewidth=1,
               label="Break-even equity for a pot-size bet (50%)")
    ax.set_xlabel("Equity")
    ax.set_ylabel("Optimal fraction of bankroll to stake (Kelly)")
    ax.set_title("Kelly stake vs. equity, facing a pot-size bet")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "kelly_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating equity comparison chart...")
    chart_equity_comparison()

    print("Training GTO solver for strategy chart...")
    solver = KuhnCFRSolver(seed=1)
    solver.train(200_000)
    chart_gto_strategy(solver)

    print("Generating CFR convergence chart (re-trains at several checkpoints)...")
    chart_cfr_convergence()

    print("Generating Kelly stake curve...")
    chart_kelly_curve()

    print(f"\nAll charts saved to: {OUT_DIR}")
    