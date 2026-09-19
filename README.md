# PokerCRR

Applying Cox-Ross-Rubinstein (CRR) binomial option-pricing logic to poker decision-making.

## Idea

A poker hand moves through discrete steps (preflop → flop → turn → river), the same way a CRR binomial tree moves through discrete time steps to price an option. This project treats poker decisions the same way a derivatives desk treats an option: equity plays the role of the underlying's price, and betting decisions are priced using no-arbitrage logic.

## Modules

- **`Equity/monte_carlo.py`** — Monte Carlo simulation (20,000 runouts) to estimate win probability at any point in a hand. Verified against known benchmarks (e.g. AA preflop heads-up ≈ 85%).
- **`Pricing/martingale.py`** — pot odds, expected value of a call, and the poker analogue of a no-arbitrage "fair" bet size.
- **`Gto/kuhn_solver.py`** — a Kuhn Poker solver trained via Counterfactual Regret Minimization (CFR), a self-play algorithm that converges to the game's known Nash equilibrium (game value = -1/18, ~1/3 Jack-bluffing frequency) without the answer being hard-coded.
- **`Bankroll/kelly.py`** — Kelly Criterion bankroll sizing, converting a known edge into an optimal stake.
- **`main.py`** — connects all four modules into one decision pipeline.
- **`visualize.py`** — generates the charts in `/charts`.
- **`generate_report.py`** — builds `PokerCRR_Report.pdf`, a full write-up with methodology, results, limitations, and code excerpts.

## Key results

- CFR converges to the theoretical Kuhn Poker game value of **-1/18**, confirming the solver actually finds the equilibrium through self-play rather than an arbitrary fixed point.
- A real bug was found and fixed during development: an early version of the Kelly module double-counted the returned stake, shifting the break-even equity point to 33% instead of the correct 50% — caught because a chart didn't match known theory.

## Running it

```bash
pip install treys matplotlib reportlab
python main.py
python visualize.py
python generate_report.py
```

## Report

See [`PokerCRR_Report.pdf`](./PokerCRR_Report.pdf) for the full write-up, including methodology, a step-by-step explanation of CFR, limitations, and code excerpts.

## Limitations

The GTO solver is exact only for Kuhn Poker (3 cards, one betting round) — real Texas Hold'em would require abstraction techniques beyond brute-force CFR. See the report's Limitations section for the full discussion.
