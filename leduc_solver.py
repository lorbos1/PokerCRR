"""
Leduc Hold'em GTO solver (Counterfactual Regret Minimization)
-------------------------------------------------------------------
Leduc Hold'em is the natural next step up from Kuhn Poker: 6 cards
(J, J, Q, Q, K, K), two betting rounds instead of one, and a public
card dealt between rounds -- the smallest standard benchmark game
that actually has a "flop" the way real poker does.

Rules:
  - 6 cards: two copies each of J, Q, K. Each player is dealt one
    private card, antes 1 chip.
  - Round 1 (preflop), bet size = 2: first-to-act can check or bet;
    facing a bet, the opponent can fold, call, or raise once (max
    one bet and one raise per round).
  - One public card is then dealt from the remaining 4 cards.
  - Round 2 (postflop), bet size = 4: same structure as round 1.
  - Showdown: a private card matching the public card's rank ("a
    pair") beats any non-pair; otherwise the higher private rank
    wins; equal non-paired ranks split the pot.

Pot contributions are tracked explicitly through the recursion (not
inferred from the action-history string), so payoffs are pot-accurate:
winning nets you exactly what your opponent contributed, losing costs
exactly what you contributed -- standard no-rake, two-player poker
accounting.

Unlike Kuhn Poker, Leduc has no simple closed-form solution here, so
this solver is validated the way real solvers are: by checking that
the estimated game value stabilizes as training iterations increase,
rather than against a specific memorized published number.
"""

import random

DECK = [0, 0, 1, 1, 2, 2]  # rank ids, two of each (J=0, Q=1, K=2)
ACTIONS_NO_BET = ["x", "b"]           # check, bet
ACTIONS_FACING_BET = ["f", "c", "r"]  # fold, call, raise
MAX_RAISES_PER_ROUND = 1
BET_SIZE = {0: 2, 1: 4}  # round_num -> bet size


class InfoSet:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.regret_sum = [0.0] * n_actions
        self.strategy_sum = [0.0] * n_actions

    def get_strategy(self, realization_weight):
        strategy = [max(r, 0.0) for r in self.regret_sum]
        total = sum(strategy)
        if total > 0:
            strategy = [s / total for s in strategy]
        else:
            strategy = [1.0 / self.n_actions] * self.n_actions
        for a in range(self.n_actions):
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / self.n_actions] * self.n_actions


def hand_rank(private, public):
    """Higher is better. A pair with the public card beats any non-pair."""
    if private == public:
        return 100 + private
    return private


class LeducCFRSolver:
    def __init__(self, seed=None):
        self.node_map = {}
        if seed is not None:
            random.seed(seed)

    def train(self, iterations):
        total_util = 0.0
        for _ in range(iterations):
            deck = DECK[:]
            random.shuffle(deck)
            cards = deck[:2]
            public_pool = deck[2:]
            total_util += self._cfr(
                cards, public_pool, round_num=0, public=None, history="",
                raises_this_round=0, contrib=(1, 1), p0=1.0, p1=1.0,
            )
        return total_util / iterations

    def _round_over(self, history):
        return history.endswith("xx") or history.endswith("c")

    def _legal_actions(self, history):
        if history == "" or history.endswith("x"):
            return ACTIONS_NO_BET
        return ACTIONS_FACING_BET

    def _cfr(self, cards, public_pool, round_num, public, history,
             raises_this_round, contrib, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # --- Fold: the player who just acted (=opponent, by the
        # negamax convention below) folded; 'player' wins opponent's
        # contribution. ---
        if history.endswith("f"):
            return float(contrib[opponent])

        # --- Round transition / showdown ---
        if self._round_over(history):
            if round_num == 0:
                new_public = public_pool[0]
                return self._cfr(
                    cards, public_pool[1:], round_num=1, public=new_public,
                    history="", raises_this_round=0, contrib=contrib, p0=p0, p1=p1,
                )
            else:
                r0 = hand_rank(cards[0], public)
                r1 = hand_rank(cards[1], public)
                # contrib[0] == contrib[1] guaranteed at true showdown
                pot_each = contrib[0]
                if r0 == r1:
                    return 0.0
                winner_is_0 = r0 > r1
                # Return from 'player' perspective (whoever's notional
                # turn it is at this terminal node, per negamax).
                if (winner_is_0 and player == 0) or (not winner_is_0 and player == 1):
                    return float(pot_each)
                else:
                    return float(-pot_each)

        legal = self._legal_actions(history)
        info_key = f"{cards[player]}{'-' if public is None else public}{round_num}{history}"
        info_set = self.node_map.setdefault(info_key, InfoSet(len(legal)))

        strategy = info_set.get_strategy(p0 if player == 0 else p1)

        util = [0.0] * len(legal)
        node_util = 0.0
        for i, a in enumerate(legal):
            effective_a = a
            next_raises = raises_this_round
            new_contrib = list(contrib)

            if a == "x":
                pass  # no change
            elif a == "b":
                new_contrib[player] += BET_SIZE[round_num]
            elif a == "f":
                pass  # payoff handled at terminal check using existing contrib
            elif a == "c":
                new_contrib[player] = contrib[opponent]
            elif a == "r":
                if raises_this_round >= MAX_RAISES_PER_ROUND:
                    # Not actually legal -- treat as a call instead.
                    effective_a = "c"
                    new_contrib[player] = contrib[opponent]
                else:
                    new_contrib[player] = contrib[opponent] + BET_SIZE[round_num]
                    next_raises = raises_this_round + 1

            next_history = history + effective_a

            if player == 0:
                util[i] = -self._cfr(
                    cards, public_pool, round_num, public, next_history,
                    next_raises, tuple(new_contrib), p0 * strategy[i], p1,
                )
            else:
                util[i] = -self._cfr(
                    cards, public_pool, round_num, public, next_history,
                    next_raises, tuple(new_contrib), p0, p1 * strategy[i],
                )
            node_util += strategy[i] * util[i]

        opponent_reach = p1 if player == 0 else p0
        for i in range(len(legal)):
            info_set.regret_sum[i] += opponent_reach * (util[i] - node_util)

        return node_util

    def print_sample_strategy(self, n=15):
        print(f"{'Info set':<22} {'Strategy (avg)'}")
        keys = sorted(self.node_map.keys())
        for i, key in enumerate(keys):
            if i >= n:
                print(f"... and {len(keys) - n} more information sets")
                break
            avg = self.node_map[key].get_average_strategy()
            print(f"{key:<22} {[round(x, 3) for x in avg]}")


if __name__ == "__main__":
    print("Training Leduc Hold'em via CFR at increasing iteration counts,")
    print("checking that the estimated game value stabilizes (the standard")
    print("convergence signature for a solver with no simple closed form):\n")

    for iters in [5_000, 50_000, 200_000, 500_000]:
        solver = LeducCFRSolver(seed=1)
        value = solver.train(iters)
        print(f"Iterations: {iters:>7}  |  Game value for P0: {value:+.4f}  |  "
              f"Information sets: {len(solver.node_map)}")

    print("\nSample of learned strategies (info set key = card, public card,")
    print("round, action history so far):")
    solver.print_sample_strategy(12)
    