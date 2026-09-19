"""
Kuhn Poker GTO solver (Counterfactual Regret Minimization)
-------------------------------------------------------------
Kuhn Poker is a simplified 3-card poker game (J, Q, K) with a known,
provably optimal (Nash equilibrium) strategy -- famous for the result
that a rational player should bluff with the worst hand (Jack) with
probability 1/3.

This is the poker analogue of solving an option by backward induction
through a binomial tree: instead of hand-coding the known closed-form
answer, we solve for it via self-play, using Counterfactual Regret
Minimization (CFR). CFR repeatedly plays the game against itself,
tracks "regret" for each action at each decision point (information
set), and provably converges to a Nash equilibrium as the number of
iterations grows -- the game-theoretic equivalent of an equilibrium
price being discovered through repeated no-arbitrage trading.

Rules (one betting round, antes of 1 each):
  - 3 cards, J < Q < K. Each player is dealt one card, ante 1 each.
  - Player 1 acts first: pass (check) or bet (bet 1).
  - If P1 passes, P2 can pass (showdown) or bet.
      - If P2 bets, P1 can pass (fold, P2 wins) or bet (call, showdown).
  - If P1 bets, P2 can pass (fold, P1 wins) or bet (call, showdown).

Action encoding: 'p' = pass/check/fold, 'b' = bet/call.
"""

import random

ACTIONS = ["p", "b"]
NUM_ACTIONS = 2


class InformationSet:
    """
    One node in the game tree, keyed by (player's own card, betting
    history so far). Tracks cumulative regret for each action, and
    the running average strategy -- CFR's output is this average,
    not the strategy at any single iteration.
    """

    def __init__(self):
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def get_strategy(self, realization_weight):
        """
        Regret matching: play each action in proportion to its
        positive regret (i.e. how much better it would have done),
        or uniformly if no action has positive regret yet.
        """
        strategy = [max(r, 0.0) for r in self.regret_sum]
        total = sum(strategy)
        if total > 0:
            strategy = [s / total for s in strategy]
        else:
            strategy = [1.0 / NUM_ACTIONS] * NUM_ACTIONS

        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self):
        """The actual equilibrium estimate: time-averaged strategy."""
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


class KuhnCFRSolver:
    def __init__(self, seed=None):
        self.node_map = {}
        if seed is not None:
            random.seed(seed)

    def train(self, iterations):
        cards = [0, 1, 2]  # 0=Jack, 1=Queen, 2=King
        util = 0.0
        for _ in range(iterations):
            random.shuffle(cards)
            util += self._cfr(cards, "", 1.0, 1.0)
        return util / iterations  # converges to the known game value -1/18

    def _cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # --- Terminal node payoffs ---
        if plays > 1:
            player_higher = cards[player] > cards[opponent]
            if history[-1] == "p":
                if history == "pp":
                    return 1.0 if player_higher else -1.0
                else:
                    # opponent folded to a bet -> current player wins the pot
                    return 1.0
            elif history[-2:] == "bb":
                return 2.0 if player_higher else -2.0

        info_set_key = f"{cards[player]}{history}"
        info_set = self.node_map.setdefault(info_set_key, InformationSet())

        strategy = info_set.get_strategy(p0 if player == 0 else p1)

        util = [0.0] * NUM_ACTIONS
        node_util = 0.0
        for a in range(NUM_ACTIONS):
            next_history = history + ACTIONS[a]
            if player == 0:
                util[a] = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self._cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        opponent_reach = p1 if player == 0 else p0
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            info_set.regret_sum[a] += opponent_reach * regret

        return node_util

    def print_strategy(self):
        card_name = {"0": "J", "1": "Q", "2": "K"}
        print(f"{'Info set':<10} {'Pass':>8} {'Bet':>8}")
        for key in sorted(self.node_map.keys(), key=lambda k: (len(k), k)):
            card, history = key[0], key[1:]
            label = card_name[card] + (f" | {history}" if history else "")
            avg = self.node_map[key].get_average_strategy()
            print(f"{label:<10} {avg[0]:>8.3f} {avg[1]:>8.3f}")


if __name__ == "__main__":
    solver = KuhnCFRSolver(seed=1)
    iterations = 200_000
    game_value = solver.train(iterations)

    print(f"Average game value for Player 1 (should approach -1/18 = -0.0556): {game_value:.4f}")
    print()
    solver.print_strategy()

    print()
    print("Known theoretical result: Player 1 bets/bluffs Jack (the worst")
    print("hand) roughly 1/3 of the time in the opening node ('J' row,")
    print("'Bet' column) -- this should emerge from training, not be")
    print("hand-coded, and is the same equilibrium first solved by Kuhn (1950).")