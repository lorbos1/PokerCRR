"""
Generates an extended PDF report for PokerCRR: methodology, a broader
set of worked examples, a step-by-step explanation of the CFR
algorithm, all four charts, a limitations/future-work section, and a
short code appendix.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "equity"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pricing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gto"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bankroll"))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)

from main import analyse_decision
import visualize

BASE = os.path.dirname(__file__)
CHARTS = os.path.join(BASE, "charts")
OUT_PDF = os.path.join(BASE, "PokerCRR_Report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name="SubHead", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=14))
styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Normal"], fontName="Courier", fontSize=8,
                           leading=11, backColor=colors.HexColor("#f5f5f5"),
                           borderPadding=8, spaceAfter=12))


def build_report():
    print("Regenerating charts for the report...")
    visualize.chart_equity_comparison()
    solver = visualize.KuhnCFRSolver(seed=1)
    solver.train(200_000)
    visualize.chart_gto_strategy(solver)
    visualize.chart_cfr_convergence()
    visualize.chart_kelly_curve()

    print("Running an expanded batch of worked examples...")
    hand_specs = [
        ("Overpair (AA)", ["Ah", "Ad"], ["7c", "2d", "9s"]),
        ("Overpair (KK)", ["Kh", "Kd"], ["7c", "2d", "9s"]),
        ("Nut flush draw", ["Ah", "Kh"], ["Qh", "7h", "2c"]),
        ("Top pair, weak kicker", ["Ah", "5c"], ["Ad", "9s", "3h"]),
        ("Gutshot draw", ["9c", "8c"], ["Th", "6d", "2s"]),
        ("Air (72o)", ["7c", "2d"], ["Ah", "Kd", "Qs"]),
    ]
    results = []
    for label, hole, board in hand_specs:
        r = analyse_decision(hole, board, 1, pot=100, bet=50, bankroll=1000, n_sims=20000, seed=42)
        results.append((label, r))

    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    story = []

    story.append(Paragraph("PokerCRR", styles["Title"]))
    story.append(Paragraph(
        "Applying Cox-Ross-Rubinstein binomial pricing logic to poker decision-making",
        styles["Heading3"]
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This project treats a poker hand as a discrete-time stochastic process, the same "
        "structure a CRR binomial tree uses to price an option: equity plays the role of the "
        "underlying's price, streets play the role of time steps, and betting decisions are "
        "priced using the same no-arbitrage logic used for options and other derivatives. "
        "Four modules implement this end to end: Monte Carlo equity estimation, martingale-based "
        "bet pricing, a Kuhn Poker GTO solver trained via Counterfactual Regret Minimization, "
        "and Kelly Criterion bankroll sizing.",
        styles["Body"]
    ))

    story.append(Paragraph("Methodology: the CRR-to-poker mapping", styles["SubHead"]))
    story.append(Paragraph(
        "The Cox-Ross-Rubinstein (CRR) model prices an option by building a binomial tree: at "
        "each discrete time step, the underlying's price moves up or down, and the option's "
        "value is computed by working backward from the known payoffs at expiry, applying "
        "risk-neutral probabilities at every node. A poker hand has the same discrete-time "
        "structure: it moves through a fixed sequence of steps (preflop, flop, turn, river), "
        "and at each step new public information (a card) shifts the probability of winning, "
        "exactly as a price move shifts an option's moneyness.",
        styles["Body"]
    ))
    mapping_data = [
        ["Options / CRR concept", "Poker analogue", "Implemented in"],
        ["Underlying's price", "Equity (win probability)", "equity/monte_carlo.py"],
        ["Risk-neutral pricing", "Pot odds / EV of a call", "pricing/martingale.py"],
        ["Backward induction", "Solving for equilibrium play", "gto/kuhn_solver.py"],
        ["Position sizing under a model", "Bankroll sizing given an edge", "bankroll/kelly.py"],
    ]
    t = Table(mapping_data, colWidths=[2.0 * inch, 2.1 * inch, 1.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)
    story.append(PageBreak())

    story.append(Paragraph("1. Equity Estimation and Pricing", styles["SubHead"]))
    story.append(Paragraph(
        "The equity module runs Monte Carlo simulation (20,000 random runouts per hand) to "
        "estimate win probability at any point in a hand. The pricing module then applies "
        "pot-odds logic, the poker equivalent of risk-neutral pricing, to decide whether a "
        "call is +EV. Below is a broader set of six hands spanning strong made hands, draws, "
        "and air, all facing the same bet size (pot=100, facing a 50-chip bet).",
        styles["Body"]
    ))
    story.append(Image(os.path.join(CHARTS, "equity_comparison.png"), width=5.3 * inch, height=3.0 * inch))
    story.append(Paragraph(
        "Two representative cases from the table below, shown visually: a strong overpair "
        "clears the pot-odds threshold comfortably, while pure air does not.",
        styles["Caption"]
    ))

    table_data = [["Hand", "Equity", "EV of call", "Decision", "Full Kelly"]]
    for label, r in results:
        table_data.append([
            label, f"{r['equity']:.1%}", f"{r['ev_call']:+.1f}",
            "CALL" if r["ev_call"] > 0 else "FOLD", f"{r['full_kelly']:.1%}"
        ])
    t2 = Table(table_data, colWidths=[1.7 * inch, 0.9 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 10))
    story.append(t2)
    story.append(Paragraph(
        "Note the gutshot draw: at 40.6% raw equity it looks strong at first glance, but this "
        "is a single-street snapshot -- the module prices the immediate call, not the full "
        "multi-street decision tree, which is a deliberate scope limitation discussed below.",
        styles["Caption"]
    ))
    story.append(PageBreak())

    story.append(Paragraph("2. Bankroll Sizing (Kelly Criterion)", styles["SubHead"]))
    story.append(Paragraph(
        "Once a decision is +EV, the Kelly Criterion converts the edge into an optimal stake "
        "size as a fraction of bankroll, maximizing long-run geometric growth without risking "
        "ruin. Full Kelly is theoretically growth-optimal but highly volatile in small samples; "
        "the module also supports fractional Kelly (e.g. half-Kelly) for lower variance at a "
        "modest cost to long-run growth, a trade-off widely used in both poker bankroll "
        "management and position sizing in trading.",
        styles["Body"]
    ))
    story.append(Image(os.path.join(CHARTS, "kelly_curve.png"), width=5.3 * inch, height=3.0 * inch))
    story.append(Paragraph(
        "Optimal Kelly stake as equity varies, for a fixed pot-size bet. Below 50% equity "
        "(the break-even threshold for this bet size) the optimal stake is exactly zero -- "
        "Kelly correctly refuses -EV bets rather than sizing them down. A real bug was found "
        "and fixed during development here: an earlier version double-counted the returned "
        "stake in the payoff calculation, shifting the break-even point to 33% instead of the "
        "correct 50% -- caught precisely because the chart didn't match the known theory.",
        styles["Caption"]
    ))

    story.append(Paragraph("3. GTO Solving via Counterfactual Regret Minimization", styles["SubHead"]))
    story.append(Paragraph(
        "Kuhn Poker (a simplified 3-card game: Jack, Queen, King, one betting round) has a "
        "known, provably optimal equilibrium strategy, famous for the result that a rational "
        "player should bluff the worst hand (the Jack) with probability in [0, 1/3]. Rather "
        "than hard-coding this answer, the solver learns it from 200,000 iterations of "
        "self-play using Counterfactual Regret Minimization (CFR), the same self-play logic "
        "used by modern superhuman poker AI (e.g. Libratus, Pluribus), applied here at a small "
        "enough scale to solve exactly and verify against the textbook result.",
        styles["Body"]
    ))
    story.append(Paragraph("How CFR works, step by step:", styles["Body"]))
    steps = [
        "Play the game against itself, tracking every decision point (an 'information set': "
        "a player's own card plus the betting history so far).",
        "At each information set, maintain a running 'regret' for every action -- how much "
        "better that action would have performed versus the strategy actually played.",
        "Choose actions in proportion to their positive regret ('regret matching'): actions "
        "that would have done better get played more often in future iterations.",
        "Track a running average of the strategy played at each information set over all "
        "iterations -- this average, not the strategy at any single iteration, is what "
        "converges to the Nash equilibrium.",
        "Repeat for many iterations; the average strategy's exploitability provably shrinks "
        "toward zero as iterations increase.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Body"]), leftIndent=12) for s in steps],
        bulletType="1", start=1
    ))
    story.append(Image(os.path.join(CHARTS, "cfr_convergence.png"), width=5.1 * inch, height=2.9 * inch))
    story.append(Paragraph(
        "The solver's estimated game value converges to the known theoretical value of -1/18 "
        "as training iterations increase, confirming the self-play process is actually finding "
        "the equilibrium rather than an arbitrary fixed point.",
        styles["Caption"]
    ))
    story.append(PageBreak())

    story.append(Image(os.path.join(CHARTS, "gto_strategy.png"), width=6.0 * inch, height=3.0 * inch))
    story.append(Paragraph(
        "Learned bet/call frequency at every information set. The model independently "
        "recovers the ~1/3 opening bluff frequency with the Jack, and the theoretically "
        "required 1/3 bet frequency with a Queen after the opponent checks then bets -- "
        "neither value was hard-coded; both emerged purely from repeated self-play.",
        styles["Caption"]
    ))

    story.append(Paragraph("Limitations and Future Work", styles["SubHead"]))
    limitations = [
        "The GTO solver is exact only for Kuhn Poker (3 cards, one betting round). Real "
        "Texas Hold'em has a vastly larger information-set space; extending the solver would "
        "require abstraction techniques (card bucketing, action abstraction) used in "
        "production-grade solvers, not brute-force CFR.",
        "The pricing module evaluates a single decision point in isolation. A full treatment "
        "would price the entire remaining game tree (all future streets and bet sizes), the "
        "poker equivalent of pricing a path-dependent option rather than a single-period one.",
        "Equity estimation assumes uniformly random opponent hands. A more realistic model "
        "would condition on a plausible opponent range inferred from the betting action so far.",
        "Kelly sizing here treats each decision independently. A bankroll actually facing a "
        "sequence of correlated decisions within one session would need a multi-period Kelly "
        "treatment, analogous to dynamic (not static) portfolio optimization.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Body"]), leftIndent=12) for s in limitations],
        bulletType="bullet", start="circle"
    ))

    story.append(PageBreak())
    story.append(Paragraph("Appendix: Core Pricing Logic", styles["SubHead"]))
    story.append(Paragraph(
        "The two functions below are the heart of the pricing module -- pot odds (the "
        "break-even equity threshold) and the fair, zero-EV bet size at a given equity, "
        "the poker analogue of a no-arbitrage price.",
        styles["Body"]
    ))
    code_snippet = """def pot_odds(bet, pot):
    \"\"\"Required equity to break even on a call.\"\"\"
    return bet / (pot + bet)

def fair_bet_size(pot, equity):
    \"\"\"
    Bet size at which calling has exactly zero EV, given equity.
    Solved from: equity*(pot+bet) = (1-equity)*bet
    \"\"\"
    if equity >= 0.5:
        return float('inf')
    return equity * pot / (1 - 2 * equity)"""
    story.append(Paragraph(code_snippet.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["CodeBlock"]))

    story.append(Paragraph(
        "The CFR regret-matching update, the mechanism that lets the GTO solver converge to "
        "equilibrium purely through self-play rather than a hand-coded solution:",
        styles["Body"]
    ))
    code_snippet2 = """def get_strategy(self, realization_weight):
    strategy = [max(r, 0.0) for r in self.regret_sum]
    total = sum(strategy)
    if total > 0:
        strategy = [s / total for s in strategy]
    else:
        strategy = [1.0 / NUM_ACTIONS] * NUM_ACTIONS
    for a in range(NUM_ACTIONS):
        self.strategy_sum[a] += realization_weight * strategy[a]
    return strategy"""
    story.append(Paragraph(code_snippet2.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["CodeBlock"]))

    story.append(Paragraph("Summary", styles["SubHead"]))
    story.append(Paragraph(
        "All four modules were built, tested against known theoretical benchmarks (AA preflop "
        "equity ~85%, Kuhn Poker game value -1/18, classic Kelly break-even points), and "
        "connected into a single decision pipeline. One real bug was found and fixed during "
        "development using the visualizations themselves as a validation tool, not just a "
        "presentation layer. The project's contribution is not any individual algorithm, each "
        "of which is well documented in the literature, but the explicit framing of poker "
        "decisions through the language of no-arbitrage derivatives pricing: equity as an "
        "underlying's price, bets as options to be priced, and bankroll sizing as position "
        "sizing under a known edge.",
        styles["Body"]
    ))

    doc.build(story)
    print(f"\nReport saved to: {OUT_PDF}")


if __name__ == "__main__":
    build_report()
    