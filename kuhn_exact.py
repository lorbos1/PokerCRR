"""
Exact Kuhn Poker equilibrium (closed form) vs. CFR-learned strategy
----------------------------------------------------------------------
Kuhn (1950) proved a closed-form family of Nash equilibria for this
game, parametrized by alpha in [0, 1/3] (Player 1's bluff frequency
with the Jack). Every other equilibrium frequency in the game is a
fixed function of alpha. This module computes those exact frequencies
and compares them, node by node, against what the CFR solver in
kuhn_solver.py actually learned -- a much stronger validation than
"the game value looks about right."

Reference: H.W. Kuhn, "A Simplified Two-Person Poker", 1950.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gto"))
from kuhn_solver import KuhnCFRSolver


def exact_strategy(alpha):
    """
    Returns exact equilibrium bet/call probabilities for every
    information set, as a function of alpha (P1's root bluff freq
    with the Jack). Keys match kuhn_solver's node_map keys
    (card + history), values are P(bet or call) -- i.e. action index 1.
    """
    if not (0 <= alpha <= 1 / 3):
        raise ValueError("alpha must be in [0, 1/3] for a valid equilibrium")

    return {
        # Player 1, root (history=""): bet frequency
        "0": alpha,              # J: bet with prob alpha
        "1": 0.0,                # Q: never bets first
        "2": min(3 * alpha, 1.0),  # K: bet with prob 3*alpha

        # Player 2, after P1 checks (history="p"): bet frequency
        "0p": 1 / 3,              # J: bet (bluff) 1/3 of the time
        "1p": 0.0,                # Q: never bets
        "2p": 1.0,                # K: always bets

        # Player 2, facing a bet (history="b"): call frequency
        "0b": 0.0,                # J: never calls
        "1b": 1 / 3,              # Q: calls 1/3 of the time
        "2b": 1.0,                # K: always calls

        # Player 1, checked then faced a bet (history="pb"): call frequency
        "0pb": 0.0,                     # J: never calls
        "1pb": min(alpha + 1 / 3, 1.0),  # Q: calls alpha + 1/3 of the time
        "2pb": 1.0,                     # K: always calls
    }


def compare(solver, alpha=None):
    """
    Compare the CFR solver's learned average strategy against the
    exact theoretical strategy. If alpha is not given, it is read
    directly from the solver's own learned P(bet | Jack, root) --
    i.e. we let the data pick which member of the equilibrium family
    to compare against, since CFR can converge to any valid alpha.
    """
    if alpha is None:
        alpha = solver.node_map["0"].get_average_strategy()[1]
        alpha = max(0.0, min(alpha, 1 / 3))

    theory = exact_strategy(alpha)

    print(f"Comparing against alpha = {alpha:.4f} (CFR's own learned root bluff frequency)\n")
    print(f"{'Info set':<10} {'CFR (learned)':>14} {'Theory (exact)':>15} {'Abs. error':>11}")

    errors = []
    for key in sorted(theory.keys(), key=lambda k: (len(k), k)):
        cfr_prob = solver.node_map[key].get_average_strategy()[1]
        theory_prob = theory[key]
        err = abs(cfr_prob - theory_prob)
        errors.append(err)
        print(f"{key:<10} {cfr_prob:>14.4f} {theory_prob:>15.4f} {err:>11.4f}")

    mae = sum(errors) / len(errors)
    max_err = max(errors)
    print(f"\nMean absolute error across all information sets: {mae:.4f}")
    print(f"Max absolute error: {max_err:.4f}")
    return mae, max_err


if __name__ == "__main__":
    print("Training CFR solver (500,000 iterations for tighter convergence)...\n")
    solver = KuhnCFRSolver(seed=1)
    solver.train(500_000)

    mae, max_err = compare(solver)

    print("\n" + "=" * 60)
    if max_err < 0.02:
        print("PASS: CFR strategy matches the exact theoretical equilibrium")
        print(f"      to within {max_err:.4f} at every information set.")
    else:
        print("Some information sets show larger deviation -- likely needs")
        print("more training iterations, or check for an implementation bug.")
        