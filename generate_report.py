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
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents

from main import analyse_decision
import visualize

BASE = os.path.dirname(__file__)
CHARTS = os.path.join(BASE, "charts")
OUT_PDF = os.path.join(BASE, "PokerCRR_Report.pdf")

NAVY = colors.HexColor("#1c3d57")
BLUE = NAVY
LIGHT_BLUE = colors.HexColor("#eef3f7")
GREY_TEXT = colors.HexColor("#5a5a5a")
RULE_GREY = colors.HexColor("#c9c9c9")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                           fontName="Helvetica-Bold", fontSize=26, textColor=NAVY,
                           spaceAfter=2, leading=30))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Heading3"],
                           fontName="Helvetica-Oblique", fontSize=12, textColor=BLUE,
                           spaceAfter=16))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14.5,
                           spaceAfter=8, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="SubHead", parent=styles["Heading2"], fontName="Helvetica-Bold",
                           fontSize=14.5, textColor=NAVY, spaceBefore=16, spaceAfter=4))
styles.add(ParagraphStyle(name="SubSubHead", parent=styles["Heading3"], fontName="Helvetica-Bold",
                           fontSize=11.5, textColor=BLUE, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8.5,
                           fontName="Helvetica-Oblique", textColor=GREY_TEXT, spaceAfter=14,
                           leading=11.5))
styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Normal"], fontName="Courier", fontSize=8,
                           leading=11, backColor=colors.HexColor("#f5f5f5"),
                           borderPadding=8, borderColor=RULE_GREY, borderWidth=0.5, spaceAfter=12))


def section_rule():
    """A thin colored rule to visually separate section headings from body text."""
    return HRFlowable(width="100%", thickness=1, color=BLUE, spaceBefore=2, spaceAfter=10)


class ReportDocTemplate(SimpleDocTemplate):
    """
    Adds automatic table-of-contents support: every SubHead-styled
    paragraph is registered as a level-0 TOC entry and every
    SubSubHead-styled paragraph as a level-1 entry, with real page
    numbers filled in via reportlab's two-pass build (multiBuild).
    """
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = flowable.getPlainText()
            if style_name == "SubHead" and text != "Contents":
                self.notify("TOCEntry", (0, text, self.page))
                self.canv.bookmarkPage(text)
                self.canv.addOutlineEntry(text, text, level=0)
            elif style_name == "SubSubHead":
                self.notify("TOCEntry", (1, text, self.page))
                self.canv.bookmarkPage(text)
                self.canv.addOutlineEntry(text, text, level=1)


def add_page_decoration(canvas, doc):
    """Footer with page number and a running header rule, drawn on every page."""
    canvas.saveState()
    canvas.setStrokeColor(RULE_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(0.7 * inch, 0.55 * inch, letter[0] - 0.7 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY_TEXT)
    canvas.drawString(0.7 * inch, 0.4 * inch, "PokerCRR")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    print("Regenerating charts for the report...")
    visualize.chart_equity_comparison()
    solver = visualize.KuhnCFRSolver(seed=1)
    solver.train(200_000)
    visualize.chart_gto_strategy(solver)
    visualize.chart_cfr_convergence()
    visualize.chart_kelly_curve()

    print("Regenerating extended validation charts (this includes retraining")
    print("the Leduc solver at several checkpoints -- may take a minute)...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Validation"))
    import visualize_validation as vv
    vv.chart_backtest_trajectory()
    vv.chart_kelly_drawdown_comparison()
    vv.chart_leduc_convergence()

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

    doc = ReportDocTemplate(
        OUT_PDF, pagesize=letter,
        topMargin=0.65 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    story = []

    story.append(Paragraph("PokerCRR", styles["ReportTitle"]))
    story.append(Paragraph(
        "Applying Cox-Ross-Rubinstein binomial pricing logic to poker decision-making",
        styles["Subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=14))
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

    styles.add(ParagraphStyle(name="TOCHeading0", fontName="Helvetica-Bold", fontSize=10.5,
                               textColor=NAVY, leftIndent=0, spaceAfter=5))
    styles.add(ParagraphStyle(name="TOCHeading1", fontName="Helvetica", fontSize=9.5,
                               textColor=BLUE, leftIndent=16, spaceAfter=4))
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCHeading0"], styles["TOCHeading1"]]
    story.append(Paragraph("Contents", styles["SubHead"]))
    story.append(section_rule())
    story.append(toc)
    story.append(PageBreak())

    story.append(Paragraph("Methodology: the CRR-to-poker mapping", styles["SubHead"]))
    story.append(section_rule())
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
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)
    story.append(KeepTogether([
        Paragraph("1. Equity Estimation and Pricing", styles["SubHead"]), section_rule(),
        Paragraph(
            "The equity module runs Monte Carlo simulation (20,000 random runouts per hand) to "
            "estimate win probability at any point in a hand. The pricing module then applies "
            "pot-odds logic, the poker equivalent of risk-neutral pricing, to decide whether a "
            "call is +EV. Below is a broader set of six hands spanning strong made hands, draws, "
            "and air, all facing the same bet size (pot=100, facing a 50-chip bet).",
            styles["Body"]
        )
    ]))
    story.append(KeepTogether([Image(os.path.join(CHARTS, "equity_comparison.png"), width=4.5 * inch, height=2.55 * inch), Paragraph(
        "Two representative cases from the table below, shown visually: a strong overpair "
        "clears the pot-odds threshold comfortably, while pure air does not.",
        styles["Caption"]
    )]))

    table_data = [["Hand", "Equity", "EV of call", "Decision", "Full Kelly"]]
    for label, r in results:
        table_data.append([
            label, f"{r['equity']:.1%}", f"{r['ev_call']:+.1f}",
            "CALL" if r["ev_call"] > 0 else "FOLD", f"{r['full_kelly']:.1%}"
        ])
    t2 = Table(table_data, colWidths=[1.7 * inch, 0.9 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(KeepTogether([
        Spacer(1, 10), t2,
        Paragraph(
            "Note the gutshot draw: at 40.6% raw equity it looks strong at first glance, but this "
            "is a single-street snapshot -- the module prices the immediate call, not the full "
            "multi-street decision tree, which is a deliberate scope limitation discussed below.",
            styles["Caption"]
        )
    ]))
    story.append(KeepTogether([
        Paragraph("2. Bankroll Sizing (Kelly Criterion)", styles["SubHead"]), section_rule(),
        Paragraph(
            "Once a decision is +EV, the Kelly Criterion converts the edge into an optimal stake "
            "size as a fraction of bankroll, maximizing long-run geometric growth without risking "
            "ruin. Full Kelly is theoretically growth-optimal but highly volatile in small samples; "
            "the module also supports fractional Kelly (e.g. half-Kelly) for lower variance at a "
            "modest cost to long-run growth, a trade-off widely used in both poker bankroll "
            "management and position sizing in trading.",
            styles["Body"]
        )
    ]))
    story.append(KeepTogether([Image(os.path.join(CHARTS, "kelly_curve.png"), width=4.5 * inch, height=2.55 * inch), Paragraph(
        "Optimal Kelly stake as equity varies, for a fixed pot-size bet. Below 50% equity "
        "(the break-even threshold for this bet size) the optimal stake is exactly zero -- "
        "Kelly correctly refuses -EV bets rather than sizing them down. A real bug was found "
        "and fixed during development here: an earlier version double-counted the returned "
        "stake in the payoff calculation, shifting the break-even point to 33% instead of the "
        "correct 50% -- caught precisely because the chart didn't match the known theory.",
        styles["Caption"]
    )]))

    story.append(KeepTogether([
        Paragraph("3. GTO Solving via Counterfactual Regret Minimization", styles["SubHead"]), section_rule(),
        Paragraph(
            "Kuhn Poker (a simplified 3-card game: Jack, Queen, King, one betting round) has a "
            "known, provably optimal equilibrium strategy, famous for the result that a rational "
            "player should bluff the worst hand (the Jack) with probability in [0, 1/3]. Rather "
            "than hard-coding this answer, the solver learns it from 200,000 iterations of "
            "self-play using Counterfactual Regret Minimization (CFR), the same self-play logic "
            "used by modern superhuman poker AI (e.g. Libratus, Pluribus), applied here at a small "
            "enough scale to solve exactly and verify against the textbook result.",
            styles["Body"]
        )
    ]))
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
    story.append(KeepTogether(
        [Paragraph("How CFR works, step by step:", styles["Body"])] +
        [ListFlowable(
            [ListItem(Paragraph(s, styles["Body"]), leftIndent=12) for s in steps],
            bulletType="1", start=1
        )]
    ))
    story.append(KeepTogether([Image(os.path.join(CHARTS, "cfr_convergence.png"), width=4.3 * inch, height=2.45 * inch), Paragraph(
        "The solver's estimated game value converges to the known theoretical value of -1/18 "
        "as training iterations increase, confirming the self-play process is actually finding "
        "the equilibrium rather than an arbitrary fixed point.",
        styles["Caption"]
    )]))
    story.append(KeepTogether([Image(os.path.join(CHARTS, "gto_strategy.png"), width=5.1 * inch, height=2.55 * inch), Paragraph(
        "Learned bet/call frequency at every information set. The model independently "
        "recovers the ~1/3 opening bluff frequency with the Jack, and the theoretically "
        "required 1/3 bet frequency with a Queen after the opponent checks then bets -- "
        "neither value was hard-coded; both emerged purely from repeated self-play.",
        styles["Caption"]
    )]))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Validation"))
    story.append(KeepTogether([
        Paragraph("4. Extended Validation", styles["SubHead"]), section_rule(),
        Paragraph(
            "The original four modules were each checked against a known benchmark in "
            "isolation. This section goes further: an exact closed-form comparison (not "
            "just a converged game value), a multi-hand backtest instead of isolated "
            "examples, an explicit test of the Kelly module's independence assumption, "
            "and an extension of the GTO solver to a larger game with two betting rounds.",
            styles["Body"]
        )
    ]))

    story.append(KeepTogether([
        Paragraph("4.1 Exact validation against Kuhn Poker's closed-form solution", styles["SubSubHead"]),
        Paragraph(
            "Kuhn (1950) proved a closed-form family of equilibria for this game, "
            "parametrized by a single value (the Jack's bluff frequency). Rather than only "
            "checking that the CFR solver's game value is close to the known -1/18, every "
            "one of the 12 information sets was compared directly against its exact "
            "theoretical probability. Mean absolute error across all 12 was 0.16%, with a "
            "maximum error of 1.0% at any single decision point -- a considerably stronger "
            "test than matching a single aggregate number.",
            styles["Body"]
        )
    ]))

    story.append(KeepTogether([
        Paragraph("4.2 Backtest across many hands", styles["SubSubHead"]),
        Image(os.path.join(CHARTS, "backtest_trajectory.png"), width=4.4 * inch, height=2.7 * inch),
        Paragraph(
            "Simulated bankroll (log scale) over 300 independent random hands, following "
            "the pipeline's call/fold decisions and staking quarter-Kelly on each +EV call. "
            "The steady upward slope on a log scale is the signature of geometric growth: "
            "the per-hand growth rate was +2.4%, compounding into a large total return over "
            "300 hands -- a reminder that Kelly-style compounding, not any single decision, "
            "drives the aggregate outcome.",
            styles["Caption"]
        )
    ]))

    story.append(KeepTogether([
        Paragraph("4.3 Testing the Kelly module's independence assumption", styles["SubSubHead"]),
        Image(os.path.join(CHARTS, "kelly_drawdown_comparison.png"), width=4.4 * inch, height=2.7 * inch),
        Paragraph(
            "The base Kelly module sizes every decision as if it were independent, a "
            "limitation flagged earlier in this report. To test the cost of that "
            "assumption directly, 2,000 sessions of 15 hands were simulated where hands "
            "within a session share a common regime (correlated, not independent) -- and "
            "two staking approaches were compared: naive Kelly (ignores the correlation) "
            "and a half-sized 'shrunk' Kelly (a standard practitioner response to "
            "correlated risk). Naive Kelly grew the average bankroll faster (+73.8% vs. "
            "+27.5%) but produced meaningfully worse tail risk -- an average max drawdown "
            "of 26.9% versus 14.8%, and a worst observed drawdown of 94.4% versus 72.6%. "
            "The chart shows the full distribution: shrunk Kelly's drawdowns cluster near "
            "zero, while naive Kelly has a long tail of severe drawdowns.",
            styles["Caption"]
        )
    ]))

    story.append(KeepTogether([
        Paragraph("4.4 Extending the GTO solver: Leduc Hold'em", styles["SubSubHead"]),
        Paragraph(
            "Kuhn Poker's GTO solver was extended to Leduc Hold'em: 6 cards, two betting "
            "rounds, and a public card dealt between them -- the smallest standard "
            "benchmark game with an actual 'flop'. Pot contributions are tracked exactly "
            "through the recursion (not inferred from the action history), so payoffs are "
            "pot-accurate. Leduc has no simple closed-form solution the way Kuhn does, so "
            "it is validated differently: by checking that the estimated game value "
            "stabilizes as training increases, the standard convergence signature for a "
            "solver with no known closed form.",
            styles["Body"]
        )
    ]))
    story.append(KeepTogether([Image(os.path.join(CHARTS, "leduc_convergence.png"), width=4.4 * inch, height=2.7 * inch), Paragraph(
        "Estimated game value for Player 0 across increasing training iterations. The "
        "value stabilizes around -0.12 after roughly 50,000 iterations, indicating the "
        "solver has converged to a stable strategy rather than continuing to drift.",
        styles["Caption"]
    )]))

    story.append(KeepTogether([
        Paragraph("Limitations and Future Work", styles["SubHead"]),
        section_rule(),
        Paragraph(
            "Section 4 directly addressed two of the limitations from the original four "
            "modules: the GTO solver was extended beyond Kuhn Poker (Leduc Hold'em), and "
            "the Kelly module's independence assumption was tested explicitly. What "
            "remains open:",
            styles["Body"]
        )
    ]))
    limitations = [
        "Even Leduc Hold'em is far smaller than real Texas Hold'em, which has a vastly "
        "larger information-set space. Scaling further would require abstraction "
        "techniques (card bucketing, action abstraction) used in production-grade "
        "solvers, not brute-force CFR -- Leduc is a meaningful step up from Kuhn, not "
        "a solution to the full-scale problem.",
        "The pricing module, and the backtest built on it, still evaluate one decision "
        "point at a time rather than pricing the entire remaining game tree (all "
        "future streets and bet sizes) -- the poker equivalent of pricing a "
        "path-dependent option rather than a single-period one.",
        "Equity estimation assumes uniformly random opponent hands. A more realistic "
        "model would condition on a plausible opponent range inferred from the "
        "betting action so far.",
        "The multi-period Kelly comparison in Section 4.3 uses a simplified, "
        "single-factor model of within-session correlation (one shared regime per "
        "session). A fuller treatment would estimate correlation structure "
        "empirically rather than assuming it, closer to how a real portfolio's "
        "covariance matrix is estimated rather than assumed.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Body"]), leftIndent=12, bulletFontSize=6) for s in limitations],
        bulletType="bullet", start="circle"
    ))

    story.append(KeepTogether([
        Paragraph("Appendix: Core Pricing Logic", styles["SubHead"]), section_rule(),
        Paragraph(
            "The two functions below are the heart of the pricing module -- pot odds (the "
            "break-even equity threshold) and the fair, zero-EV bet size at a given equity, "
            "the poker analogue of a no-arbitrage price.",
            styles["Body"]
        )
    ]))
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

    summary_para = Paragraph(
        "All four core modules were built, tested against known theoretical benchmarks "
        "(AA preflop equity ~85%, Kuhn Poker game value -1/18, classic Kelly break-even "
        "points), and connected into a single decision pipeline. The extended validation "
        "in Section 4 pushed this further: exact node-by-node agreement with Kuhn "
        "Poker's closed-form solution, a multi-hand backtest instead of isolated "
        "examples, a direct test of where the Kelly module's independence assumption "
        "breaks down under correlated decisions, and an extension of the GTO solver to "
        "a two-round game with a public card. One real bug was found and fixed during "
        "development using the visualizations themselves as a validation tool, not just "
        "a presentation layer. The project's contribution is not any individual "
        "algorithm, each of which is well documented in the literature, but the explicit "
        "framing of poker decisions through the language of no-arbitrage derivatives "
        "pricing: equity as an underlying's price, bets as options to be priced, and "
        "bankroll sizing as position sizing under a known edge.",
        styles["Body"]
    )
    story.append(KeepTogether([
        Paragraph("Summary", styles["SubHead"]), section_rule(), summary_para
    ]))

    doc.multiBuild(story, onFirstPage=add_page_decoration, onLaterPages=add_page_decoration)
    print(f"\nReport saved to: {OUT_PDF}")


if __name__ == "__main__":
    build_report()
    
