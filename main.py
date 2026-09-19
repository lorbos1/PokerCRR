"""
PokerCRR -- main workflow
---------------------------
Ties the four modules into a single decision pipeline, the same way a
CRR option-pricing workflow moves from "estimate the underlying's
distribution" to "price the derivative" to "size the position":

  1. equity/monte_carlo.py    -> estimate win probability at this node
  2. pricing/martingale.py    -> price the bet/call given that equity
  3. gto/kuhn_solver.py       -> (separate, simplified game) solve for
                                  the equilibrium strategy itself
  4. bankroll/kelly.py        -> convert edge into an optimal stake

Run this file directly to see a worked example end to end.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "equity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pricing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gto"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bankroll"))

from monte_carlo import estimate_equity
from martingale import pot_odds, ev_call, fair_bet_size, bluff_indifference_frequency
from kuhn_solver import KuhnCFRSolver
from kelly import kelly_from_pot_odds, fractional_kelly


def analyse_decision(hole_cards, board, n_opponents, pot, bet, bankroll, n_sims=20000, seed=42):
    """
    Full pipeline: equity -> pricing -> sizing, for one concrete
    decision point (a "node" in the CRR-style tree).
    """
    print("=" * 60)
    print(f"Hand: {hole_cards}  Board: {board}  Opponents: {n_opponents}")
    print(f"Pot: {pot}  Facing bet: {bet}  Bankroll: {bankroll}")
    print("=" * 60)

    # --- Step 1: Equity (Monte Carlo) ---
    result = estimate_equity(hole_cards, board=board, n_opponents=n_opponents,
                              n_sims=n_sims, seed=seed)
    equity = result["win"] + result["tie"] / 2  # count ties as half a win
    print(f"\n[1] Equity (Monte Carlo, {n_sims} sims): "
          f"win={result['win']:.3f} tie={result['tie']:.3f} lose={result['lose']:.3f}")
    print(f"    -> effective equity used downstream: {equity:.3f}")

    # --- Step 2: Pricing (martingale / no-arbitrage) ---
    required = pot_odds(bet, pot)
    ev = ev_call(pot, bet, equity)
    fair = fair_bet_size(pot, equity)
    bluff_freq = bluff_indifference_frequency(pot, bet)
    print(f"\n[2] Pricing:")
    print(f"    Pot odds require {required:.3f} equity to break even")
    print(f"    Your equity is {equity:.3f} -> EV of calling = {ev:+.2f} chips")
    print(f"    Decision: {'CALL (+EV)' if ev > 0 else 'FOLD (-EV)'}")
    print(f"    Fair (zero-EV) bet size at this equity would be: {fair:.1f}")
    print(f"    A bettor here should be bluffing {bluff_freq:.1%} of the time in equilibrium")

    # --- Step 3: Kelly sizing ---
    full_kelly = kelly_from_pot_odds(equity, pot, bet)
    half_kelly = fractional_kelly(equity, pot / bet, fraction=0.5)
    print(f"\n[3] Kelly sizing (only meaningful if the call is +EV):")
    print(f"    Full Kelly: {full_kelly:.1%} of bankroll -> {full_kelly * bankroll:.2f} chips")
    print(f"    Half Kelly (lower variance): {half_kelly:.1%} of bankroll -> {half_kelly * bankroll:.2f} chips")

    return {
        "equity": equity,
        "pot_odds_required": required,
        "ev_call": ev,
        "fair_bet": fair,
        "bluff_freq": bluff_freq,
        "full_kelly": full_kelly,
        "half_kelly": half_kelly,
    }


def run_gto_solver(iterations=200_000, seed=1):
    """Step 4 (standalone): solve Kuhn Poker via CFR self-play."""
    print("\n" + "=" * 60)
    print("Kuhn Poker GTO solver (Counterfactual Regret Minimization)")
    print("=" * 60)
    solver = KuhnCFRSolver(seed=seed)
    game_value = solver.train(iterations)
    print(f"\nGame value for Player 1 (theory: -1/18 = -0.0556): {game_value:.4f}")
    print()
    solver.print_strategy()
    return solver


if __name__ == "__main__":
    # Worked example 1: strong hand, clearly +EV call
    analyse_decision(
        hole_cards=["Kh", "Kd"],
        board=["7c", "2d", "9s"],
        n_opponents=1,
        pot=100,
        bet=50,
        bankroll=1000,
    )

    # Worked example 2: weak hand, clearly -EV call
    analyse_decision(
        hole_cards=["7c", "2d"],
        board=["Ah", "Kd", "Qs"],
        n_opponents=1,
        pot=100,
        bet=50,
        bankroll=1000,
    )

    # Step 4: the GTO solver, run separately (it solves its own simplified game)
    run_gto_solver()
    