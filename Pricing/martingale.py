"""
Martingale pricing module
--------------------------
Under risk-neutral ("martingale") pricing in finance, an asset's price
today equals its expected discounted future payoff -- there is no
systematic drift once you price correctly. In poker, the analogous
idea is pot odds: a call is "fairly priced" exactly when your equity
equals the pot odds offered. This module prices bets/calls the same
way a risk-neutral pricing desk would price a derivative payoff.
"""


def pot_odds(bet, pot):
    """
    Required equity to break even on a call.

    If pot = P (before your call) and you must call `bet` to see
    showdown, you risk `bet` to win `pot + bet`. Break-even equity
    is bet / (pot + bet) -- the classic pot-odds formula, which is
    just the risk-neutral "strike price" of the call.
    """
    return bet / (pot + bet)


def ev_call(pot, bet, equity):
    """
    Expected value of calling, in chips, given your true equity.

    EV = equity * (pot + bet) - (1 - equity) * bet
       = equity * (pot + bet) - bet + equity * bet
    A positive EV means the call is underpriced relative to your
    equity (i.e. the martingale/no-arbitrage price says: call).
    """
    win_amount = pot + bet
    return equity * win_amount - (1 - equity) * bet


def fair_bet_size(pot, equity):
    """
    Solve for the bet size at which calling has exactly zero EV,
    given a fixed equity. This is the 'martingale-fair' bet: the
    price at which the call is a fair asset (zero expected drift).

    From ev_call(pot, bet, equity) = 0:
        equity * (pot + bet) = (1 - equity) * bet
        equity * pot = bet * (1 - equity - equity)
        equity * pot = bet * (1 - 2*equity)
        bet = equity * pot / (1 - 2*equity)   [only valid for equity < 0.5]
    """
    if equity >= 0.5:
        return float("inf")  # any bet size is +EV to call; no finite fair price
    return equity * pot / (1 - 2 * equity)


def bluff_indifference_frequency(pot, bet):
    """
    In equilibrium, a bettor's bluff-to-value ratio must make the
    opponent exactly indifferent between calling and folding. This
    is the poker analogue of a no-arbitrage condition: if the bluff
    frequency deviates from this value, the opponent has a strictly
    profitable deviation (call more vs. fold more).

    Required bluff frequency (as a fraction of the betting range) is
    bet / (pot + bet) -- numerically identical to pot_odds(), since
    both come from the same zero-EV indifference condition, just
    applied to opposite sides of the bet.
    """
    return bet / (pot + bet)


if __name__ == "__main__":
    # Sanity check: classic half-pot bet -> opponent needs 33.3% equity to call
    print("Pot odds, bet=50 into pot=100:", pot_odds(50, 100))  # expect 0.333

    # If your true equity is 40%, calling a half-pot bet should be +EV
    print("EV of call, 40% equity vs half-pot bet:", ev_call(100, 50, 0.40))

    # If your true equity is only 15%, calling should be -EV
    print("EV of call, 15% equity vs half-pot bet:", ev_call(100, 50, 0.15))

    # Fair (zero-EV) bet size for someone who holds exactly 25% equity, pot=100
    print("Fair bet size, 25% equity, pot=100:", fair_bet_size(100, 0.25))

    # Bluff indifference frequency for a pot-sized bet is always 50%
    print("Bluff indifference freq, pot-size bet:", bluff_indifference_frequency(100, 100))
