"""
Kelly Criterion module
------------------------
Once you know your edge (from equity estimation and/or the GTO
solver), the Kelly Criterion answers the last question: how much of
your bankroll should you actually risk on this decision, to maximize
long-run geometric growth without risking ruin?

This is the module that turns "I have an edge" into "here is how much
to bet" -- the poker/trading equivalent of position sizing.
"""


def kelly_fraction(win_prob, win_payoff, loss_payoff=1.0):
    """
    Classic Kelly formula for a bet with two outcomes.

    Parameters
    ----------
    win_prob : float
        Probability of winning (your equity/edge).
    win_payoff : float
        Net units won per unit staked, if you win (b in "b:1 odds").
    loss_payoff : float
        Net units lost per unit staked, if you lose. Usually 1.0
        (you lose your whole stake).

    Returns
    -------
    Optimal fraction of bankroll to stake (can be negative, meaning
    "don't take this bet" -- clipped to 0 by kelly_fraction_safe).

    Formula: f* = (b*p - q) / b, where b = win_payoff, p = win_prob,
    q = 1 - p = loss_prob. Equivalently: f* = p/loss_payoff -
    q/win_payoff, generalized for asymmetric payoffs.
    """
    p = win_prob
    q = 1 - win_prob
    return (win_payoff * p - q * loss_payoff) / win_payoff


def kelly_fraction_safe(win_prob, win_payoff, loss_payoff=1.0, fraction_cap=1.0):
    """
    Same as kelly_fraction, but clipped to [0, fraction_cap]. A
    negative Kelly fraction means the bet is -EV and should be
    skipped entirely, not shorted.
    """
    f = kelly_fraction(win_prob, win_payoff, loss_payoff)
    return max(0.0, min(f, fraction_cap))


def fractional_kelly(win_prob, win_payoff, loss_payoff=1.0, fraction=0.5):
    """
    Many practitioners (in both poker and trading) bet a fixed
    fraction of full Kelly -- e.g. 'half Kelly' -- to reduce variance
    at a modest cost to long-run growth rate. This is standard
    practice because full Kelly is extremely volatile in small
    samples, even though it is growth-optimal asymptotically.
    """
    full = kelly_fraction_safe(win_prob, win_payoff, loss_payoff)
    return full * fraction


def kelly_from_pot_odds(equity, pot, bet):
    """
    Convenience wrapper connecting this module directly to the
    equity (Monte Carlo) and pricing (martingale) modules: given your
    equity and the pot/bet sizes, compute the Kelly-optimal fraction
    of bankroll to risk on a call.

    IMPORTANT: the Kelly 'b' parameter is net odds -- profit per unit
    staked if you win, NOT total return. If you call `bet` and win,
    you gain the pot that was already there (pot), not pot + bet:
    the `bet` portion is your own money returning to you, not profit.
    So b = pot / bet here, matching the standard pot-odds breakeven
    (p = bet / (pot + bet)  <=>  p = 1 / (1 + b) with b = pot/bet).
    """
    win_payoff = pot / bet
    return kelly_fraction_safe(equity, win_payoff, loss_payoff=1.0)


if __name__ == "__main__":
    # Classic sanity check: 60% win probability, even-money (1:1) bet
    # Full Kelly should say bet 20% of bankroll: f* = 2*0.6 - 1 = 0.2
    print("60% win prob, even money:", kelly_fraction(0.6, win_payoff=1.0))

    # A bet with no edge (50/50, even money) should say bet nothing
    print("50% win prob, even money (no edge):", kelly_fraction(0.5, win_payoff=1.0))

    # A clearly -EV bet should return negative, clipped to 0 by the safe version
    print("30% win prob, even money (raw):", kelly_fraction(0.3, win_payoff=1.0))
    print("30% win prob, even money (safe, clipped):", kelly_fraction_safe(0.3, win_payoff=1.0))

    # Half-Kelly for the same 60% edge
    print("60% win prob, even money, half-Kelly:", fractional_kelly(0.6, win_payoff=1.0, fraction=0.5))

    # Tie this to the poker context: AA preflop (~85% equity) facing a pot-size bet
    print(
        "Kelly fraction, 85% equity vs pot-size bet (pot=100, bet=100):",
        kelly_from_pot_odds(equity=0.85, pot=100, bet=100),
    )
    