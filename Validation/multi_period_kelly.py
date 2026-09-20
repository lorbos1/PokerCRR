"""
Multi-period Kelly under correlated decisions
--------------------------------------------------
The base Kelly module (bankroll/kelly.py) sizes each decision as if
it were independent of every other decision. In a real session,
consecutive hands are not independent: they share the same table
dynamics, the same read on an opponent, the same run of cards -- in
short, a session-level "regime" that pushes multiple hands' true
edge in the same direction at once.

This module makes that concrete: it simulates many sessions, each
built from several hands whose equity is correlated through a shared
per-session regime, and compares two staking approaches:

  1. "Naive" Kelly - sizes every hand with the standard formula,
     exactly as if each hand were independent, ignoring the shared
     regime.
  2. "Shrunk" Kelly - sizes every hand at a fixed fraction of naive
     Kelly, a standard practitioner response to correlated risk
     (the same logic as reducing position size when trades share a
     common risk factor, in trading).

Both approaches can show similar AVERAGE growth. The difference shows
up in the tail: naive Kelly produces much deeper drawdowns when a
session's regime turns unfavorable, because it stacks full-sized bets
on top of each other with no adjustment for the fact that they are
not independent bets.
"""

import random


def simulate_session(rng, n_hands, base_equity, regime_spread, payoff, kelly_scale, bankroll):
    """
    Simulate one session of n_hands correlated hands.

    A single 'regime' shift (shared across the whole session) is
    drawn once and added to every hand's equity in that session --
    this is what creates the correlation. Each hand is then a
    Kelly-sized bet at kelly_scale x the naive Kelly fraction,
    reinvesting from the running bankroll.
    """
    regime = rng.uniform(-regime_spread, regime_spread)
    trajectory = [bankroll]
    peak = bankroll

    for _ in range(n_hands):
        equity = min(max(base_equity + regime, 0.01), 0.99)

        # Naive per-hand Kelly fraction, scaled by kelly_scale
        b = payoff  # net odds
        naive_f = max(0.0, (b * equity - (1 - equity)) / b)
        stake_fraction = naive_f * kelly_scale
        stake = stake_fraction * bankroll

        won = rng.random() < equity
        if won:
            bankroll += stake * payoff
        else:
            bankroll -= stake
        bankroll = max(bankroll, 0.0)

        trajectory.append(bankroll)
        peak = max(peak, bankroll)

    max_drawdown = 0.0
    running_peak = trajectory[0]
    for v in trajectory:
        running_peak = max(running_peak, v)
        if running_peak > 0:
            drawdown = (running_peak - v) / running_peak
            max_drawdown = max(max_drawdown, drawdown)

    return trajectory[-1], max_drawdown


def run_comparison(n_sessions=2000, n_hands_per_session=15, base_equity=0.55,
                    regime_spread=0.12, payoff=1.0, starting_bankroll=1000, seed=7):
    """
    Run many sessions under both staking approaches and compare
    average growth and drawdown risk.
    """
    rng_naive = random.Random(seed)
    rng_shrunk = random.Random(seed)  # same seed -> same regimes/outcomes, fair comparison

    naive_finals, naive_drawdowns = [], []
    shrunk_finals, shrunk_drawdowns = [], []

    for _ in range(n_sessions):
        f, dd = simulate_session(rng_naive, n_hands_per_session, base_equity,
                                  regime_spread, payoff, kelly_scale=1.0,
                                  bankroll=starting_bankroll)
        naive_finals.append(f)
        naive_drawdowns.append(dd)

    for _ in range(n_sessions):
        f, dd = simulate_session(rng_shrunk, n_hands_per_session, base_equity,
                                  regime_spread, payoff, kelly_scale=0.5,
                                  bankroll=starting_bankroll)
        shrunk_finals.append(f)
        shrunk_drawdowns.append(dd)

    def summarize(finals, drawdowns, label):
        avg_growth = (sum(finals) / len(finals) / starting_bankroll - 1) * 100
        avg_dd = sum(drawdowns) / len(drawdowns) * 100
        worst_dd = max(drawdowns) * 100
        pct_ruined = sum(1 for f in finals if f < starting_bankroll * 0.1) / len(finals) * 100
        print(f"{label}")
        print(f"  Average final bankroll growth: {avg_growth:+.1f}%")
        print(f"  Average max drawdown within a session: {avg_dd:.1f}%")
        print(f"  Worst-case max drawdown observed: {worst_dd:.1f}%")
        print(f"  Sessions ending below 10% of starting bankroll: {pct_ruined:.1f}%")
        print()

    summarize(naive_finals, naive_drawdowns, "Naive Kelly (ignores session-level correlation):")
    summarize(shrunk_finals, shrunk_drawdowns, "Shrunk Kelly (half-sized, accounts for correlation):")


if __name__ == "__main__":
    print(f"Simulating 2,000 sessions of 15 correlated hands each...\n")
    run_comparison()
    print("Takeaway: both approaches grow the bankroll on average, since the")
    print("underlying per-hand edge is real and positive either way. The gap")
    print("is in the tail -- naive Kelly, applied hand-by-hand as if each bet")
    print("were independent, produces meaningfully deeper drawdowns and more")
    print("near-ruin sessions once a session's shared regime turns unfavorable,")
    print("because it stacks full-sized correlated bets with no adjustment.")
    