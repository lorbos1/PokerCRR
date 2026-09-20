# PokerCRR

Applying Cox-Ross-Rubinstein (CRR) binomial option-pricing logic to poker decision-making.

## Idea

A poker hand moves through discrete steps (preflop → flop → turn → river), the same way a CRR binomial tree moves through discrete time steps to price an option. This project treats poker decisions the same way a derivatives desk treats an option: equity plays the role of the underlying's price, and betting decisions are priced using no-arbitrage logic.

## Core modules

- **`Equity/monte_carlo.py`** — Monte Carlo simulation (20,000 runouts) to estimate win probability at any point in a hand. Verified against known benchmarks (e.g. AA preflop heads-up ≈ 85%).
- **`Pricing/martingale.py`** — pot odds, expected value of a call, and the poker analogue of a no-arbitrage "fair" bet size.
- **`Gto/kuhn_solver.py`** — a Kuhn Poker solver trained via Counterfactual Regret Minimization (CFR), converging to the game's known Nash equilibrium (game value = -1/18, ~1/3 Jack-bluffing frequency) without the answer being hard-coded.
- **`Bankroll/kelly.py`** — Kelly Criterion bankroll sizing, converting a known edge into an optimal stake.
- **`main.py`** — connects all four modules into one decision pipeline.
- **`visualize.py`** — generates the core charts in `/charts`.
- **`generate_report.py`** — builds `PokerCRR_Report.pdf`, the full write-up.

## Validation module

`/Validation` extends the rigor beyond the core modules:

- **`kuhn_exact.py`** — compares the CFR solver's learned strategy against Kuhn Poker's exact closed-form solution (Kuhn, 1950), node by node, not just the aggregate game value. Mean absolute error: 0.16%, max: 1.0%.
- **`backtest.py`** — runs the full pipeline across hundreds of random hands instead of isolated examples, tracking simulated bankroll growth.
- **`multi_period_kelly.py`** — tests the base Kelly module's independence assumption directly, comparing naive vs. correlation-adjusted staking across sessions of correlated hands.
- **`leduc_solver.py`** — extends the GTO solver beyond Kuhn Poker to Leduc Hold'em (two betting rounds, a public card), with pot-accurate payoff tracking.
- **`visualize_validation.py`** — generates the additional charts for these four extensions.

## Key results

- CFR converges to the theoretical Kuhn Poker game value of **-1/18**, and matches the exact equilibrium strategy at every information set to within 1%.
- Extending to Leduc Hold'em, the solver's estimated game value stabilizes as training increases — the standard convergence signature for a game with no simple closed form.
- Naive Kelly sizing, applied as if decisions were independent, produces materially worse drawdown risk than correlation-adjusted sizing once decisions are actually correlated (94% vs. 73% worst-case drawdown across 2,000 simulated sessions) — a concrete demonstration of why the base module's independence assumption matters.
- A real bug was found and fixed during development: an early version of the Kelly module double-counted the returned stake, shifting the break-even equity point to 33% instead of the correct 50% — caught because a chart didn't match known theory.

## Running it

```bash
pip install treys matplotlib reportlab
python main.py
python visualize.py
python Validation/kuhn_exact.py
python Validation/backtest.py
python Validation/multi_period_kelly.py
python Validation/leduc_solver.py
python generate_report.py
```

## Report

See [`PokerCRR_Report.pdf`](./PokerCRR_Report.pdf) for the full write-up: methodology, results, the extended validation, limitations, and code excerpts.

## Limitations

The GTO solver is exact only for Kuhn Poker, and Leduc Hold'em extends it to a slightly larger game — real Texas Hold'em would still require abstraction techniques beyond brute-force CFR. See the report's Limitations section for the full discussion of what each module does and does not capture.
