"""
Equity Monte Carlo module
--------------------------
Estimates a hand's equity (win probability) at any point in a hand,
given known hole cards, known board cards, and how many opponents
remain. This is the "current price" at a node of the CRR-style tree:
every other module (martingale pricing, GTO, Kelly) consumes the
equity number this module produces.

Uses the `treys` library for fast 7-card hand evaluation.
"""

import random
from treys import Card, Deck, Evaluator

evaluator = Evaluator()


def _remaining_deck(known_cards):
    """
    Return a shuffled list of card ints with all known cards removed.
    NOTE: we do NOT reuse treys.Deck.shuffle() here, since it resets
    .cards to the full 52-card deck internally (see treys source) and
    would silently undo our filtering.
    """
    full = Deck.GetFullDeck()
    remaining = [c for c in full if c not in known_cards]
    random.shuffle(remaining)
    return remaining


def estimate_equity(hole_cards, board=None, n_opponents=1, n_sims=20000, seed=None):
    """
    Estimate equity (win probability) via Monte Carlo simulation.

    Parameters
    ----------
    hole_cards : list[str]
        Your two hole cards, e.g. ["As", "Kd"] (treys string format).
    board : list[str] or None
        Community cards already revealed (0, 3, 4, or 5 cards).
    n_opponents : int
        Number of opponents still in the hand.
    n_sims : int
        Number of random runouts to simulate.
    seed : int or None
        Optional RNG seed for reproducibility.

    Returns
    -------
    dict with 'win', 'tie', 'lose' probabilities (sum to 1.0).
    """
    if seed is not None:
        random.seed(seed)

    board = board or []
    hole = [Card.new(c) for c in hole_cards]
    board_cards = [Card.new(c) for c in board]

    wins = ties = losses = 0
    cards_to_deal = 5 - len(board_cards)
    known = hole + board_cards

    for _ in range(n_sims):
        remaining = _remaining_deck(known)
        idx = 0

        # deal opponent hole cards
        opp_holes = []
        for _ in range(n_opponents):
            opp_holes.append([remaining[idx], remaining[idx + 1]])
            idx += 2

        # complete the board
        if cards_to_deal > 0:
            full_board = board_cards + remaining[idx:idx + cards_to_deal]
        else:
            full_board = board_cards

        my_score = evaluator.evaluate(full_board, hole)
        opp_scores = [evaluator.evaluate(full_board, oh) for oh in opp_holes]

        best_opp = min(opp_scores)  # treys: lower score = stronger hand

        if my_score < best_opp:
            wins += 1
        elif my_score == best_opp:
            ties += 1
        else:
            losses += 1

    total = wins + ties + losses
    return {
        "win": wins / total,
        "tie": ties / total,
        "lose": losses / total,
    }


if __name__ == "__main__":
    # Quick sanity check: pocket aces preflop vs 1 random opponent
    # should sit around ~85% equity heads-up.
    result = estimate_equity(["As", "Ad"], board=[], n_opponents=1, n_sims=20000, seed=42)
    print("AA preflop vs 1 opponent:", result)

    # Sanity check 2: weak hand postflop, board favors opponent range less directly
    result2 = estimate_equity(
        ["7c", "2d"], board=["Ah", "Kd", "Qs"], n_opponents=1, n_sims=20000, seed=42
    )
    print("72o on AKQ board vs 1 opponent:", result2)
