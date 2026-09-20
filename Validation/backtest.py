"""
Backtest: equity -> pricing -> Kelly, run across many random hands
------------------------------------------------------------------------
Earlier examples evaluated single hands in isolation. This module
generates many random hands, runs each through the full pipeline, and
tracks what would have happened to a bankroll that followed the
pipeline's decisions every time -- turning "these four functions each
work" into "here is the aggregate outcome of using them repeatedly."

This is still a simplification (each hand is treated as an
independent, single-street decision -- see the report's Limitations
section), but it is a meaningfully stronger test than isolated
examples.
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "equity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pricing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bankroll"))

from treys import Card, Deck
from monte_carlo import estimate_equity
from martingale import pot_odds, ev_call
from kelly import kelly_from_pot_odds, fractional_kelly


def random_hand(seed):
    """Deal a random hole-card pair and a random 3-card flop."""
    rng = random.Random(seed)
    deck = Deck.GetFullDeck()
    rng.shuffle(deck)
    hole = [Card.int_to_str(c) for c in deck[:2]]
    board = [Card.int_to_str(c) for c in deck[2:5]]
    return hole, board


def run_backtest(n_hands=500, pot=100, bet=50, starting_bankroll=1000,
                  kelly_fraction=0.5, n_sims=3000, seed=0):
    """
    Simulate n_hands independent hands. For each: estimate equity,
    decide call/fold via EV, and if +EV, stake fractional Kelly of
    the CURRENT bankroll on the outcome (win = +pot, lose = -bet).
    Tracks bankroll trajectory to see the net effect of always
    following the pipeline's decisions.
    """
    bankroll = starting_bankroll
    trajectory = [bankroll]
    n_calls = 0
    n_folds = 0
    n_wins = 0
    n_losses = 0

    rng = random.Random(seed)

    for i in range(n_hands):
        hole, board = random_hand(seed=seed * 100_000 + i)
        result = estimate_equity(hole, board=board, n_opponents=1, n_sims=n_sims, seed=None)
        equity = result["win"] + result["tie"] / 2
        ev = ev_call(pot, bet, equity)

        if ev <= 0:
            n_folds += 1
            trajectory.append(bankroll)  # folding costs nothing further here
            continue

        n_calls += 1
        stake_fraction = fractional_kelly(equity, pot / bet, fraction=kelly_fraction)
        stake = stake_fraction * bankroll

        # Simulate the actual outcome of the hand given true equity
        won = rng.random() < equity
        if won:
            bankroll += stake * (pot / bet)  # net profit per unit staked
            n_wins += 1
        else:
            bankroll -= stake
            n_losses += 1

        bankroll = max(bankroll, 0.0)  # bankroll cannot go negative
        trajectory.append(bankroll)

    return {
        "trajectory": trajectory,
        "final_bankroll": bankroll,
        "starting_bankroll": starting_bankroll,
        "n_hands": n_hands,
        "n_calls": n_calls,
        "n_folds": n_folds,
        "n_wins": n_wins,
        "n_losses": n_losses,
    }


if __name__ == "__main__":
    print("Running backtest across 300 random hands (pot-size bet, quarter-Kelly)...\n")
    result = run_backtest(n_hands=300, pot=100, bet=100, kelly_fraction=0.25, seed=42)

    growth = (result["final_bankroll"] / result["starting_bankroll"] - 1) * 100
    win_rate = result["n_wins"] / result["n_calls"] if result["n_calls"] else 0

    print(f"Hands played:      {result['n_hands']}")
    print(f"Calls:             {result['n_calls']} ({result['n_calls']/result['n_hands']:.1%} of hands)")
    print(f"Folds:             {result['n_folds']} ({result['n_folds']/result['n_hands']:.1%} of hands)")
    print(f"Win rate on calls: {win_rate:.1%}")
    print(f"Starting bankroll: {result['starting_bankroll']:.2f}")
    print(f"Final bankroll:    {result['final_bankroll']:.2f}")
    print(f"Net growth:        {growth:+.1f}%")
    if result["n_calls"] > 0:
        per_hand_growth = (result["final_bankroll"] / result["starting_bankroll"]) ** (1 / result["n_calls"]) - 1
        print(f"Per-hand geometric growth rate: {per_hand_growth:+.2%}")
    print()
    print("The total percentage looks extreme because Kelly-sized bankrolls")
    print("compound geometrically by design -- this is the same reason Kelly")
    print("curves are conventionally plotted on a log scale (see the chart).")
    print("The per-hand growth rate above is the more meaningful number: a")
    print("small, repeatable edge per decision, compounding over many hands.")
    print()
    print("Note: this simplified backtest treats each hand as an independent")
    print("event with a fixed pot/bet size and a known true equity used to")
    print("simulate the outcome -- it demonstrates the pipeline's aggregate")
    print("behavior, not a claim about real-world profitability (see the")
    print("report's Limitations section for the gap between this and a real")
    print("multi-street, adversarial poker session).")
    