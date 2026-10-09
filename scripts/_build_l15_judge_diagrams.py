"""Diagrams for L15 - LLM-as-a-Judge & Advanced Evaluation.

Built from scripts/diagram_kit.py. Notebook numbers are the ones verified by
scripts/verify_llm_judge_results.py. Figures that go beyond the notebook carry
an "EXTENSION" badge inside the image.

Run:  python3 scripts/_build_l15_judge_diagrams.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_kit import *   # noqa
import diagram_kit as K
import textwrap as tw

set_output(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "8. Agentic AI Systems", "15.  LLM-as-a-Judge & Advanced Evaluation", "images")))


# ------------------------------------------------------------------ helpers
def box(ax, x, y, w, h, tint, head, text="", fs=8.4, head_fs=9.6, accent=True, fill=None):
    card(ax, x, y, w, h, tint=fill, accent=tint if accent else None, z=3)
    ax.text(x + 2.0, y + h - 3.0, head, fontsize=head_fs, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    if text:
        lines = []
        for para in text.split("\n"):
            lines += tw.wrap(para, wrap_chars(ax, w - 3.6, fs)) or [""]
        lh = K._lineh(ax, fs, 1.6)
        yy = y + h - 3.0 - lh * 1.35
        for ln in lines:
            ax.text(x + 2.0, yy, ln, fontsize=fs, color=BODY, va="center",
                    fontfamily=SANS, zorder=5)
            yy -= lh
        return yy
    return y + h - 3.0


def strip(ax, x, y, w, h, tint, glyph, text, fs=8.0):
    card(ax, x, y, w, h, tint=tint, accent=tint, z=3)
    chip(ax, x + 4.0, y + h / 2, tint, glyph, r=1.7, fs=8.5)
    lines = tw.wrap(text, wrap_chars(ax, w - 11.0, fs))
    lh = K._lineh(ax, fs, 1.65)
    yy = y + h / 2 + lh * (len(lines) - 1) / 2
    for ln in lines:
        ax.text(x + 7.4, yy, ln, fontsize=fs, color=STRONG[tint], va="center",
                fontfamily=SANS, zorder=5, fontweight="bold")
        yy -= lh


def ext_badge(ax, x=96.5, y=96.0, text="EXTENSION - beyond notebook"):
    ax.text(x, y, text, fontsize=7.6, color=PURPLE, ha="right", va="center",
            fontweight="bold", fontfamily=MONO, zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", fc=TINT["purple"], ec=PURPLE, lw=0.8))


def hbar(ax, x, y, w, h, pct, tint, label_txt, value_txt, lw=24, vw=10):
    ax.text(x, y + h / 2, label_txt, fontsize=8.8, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    bx, bw = x + lw, w - lw - vw
    ax.add_patch(K.FancyBboxPatch((bx, y), bw, h, boxstyle="round,pad=0,rounding_size=0.8",
                 fc="#EDF1F7", ec="none", zorder=4))
    if pct > 0:
        ax.add_patch(K.FancyBboxPatch((bx, y), max(bw * pct / 100.0, 1.0), h,
                     boxstyle="round,pad=0,rounding_size=0.8", fc=STRONG[tint], ec="none", zorder=5))
    ax.text(x + w, y + h / 2, value_txt, fontsize=11, color=STRONG[tint], va="center",
            ha="right", fontweight="bold", fontfamily=SANS, zorder=6)


def matrix(ax, x, y, cw, ch, vals, title_txt=None, fs=15):
    """2x2 confusion matrix. vals = (a, b, c, d) = (PP, PF, FP, FF). Rows = rater A."""
    tints = ["green", "amber", "amber", "rose"]
    pos = [(0, 1), (1, 1), (0, 0), (1, 0)]
    for (i, j), v, t in zip(pos, vals, tints):
        cx, cy = x + i * (cw + 0.8), y + j * (ch + 0.8)
        card(ax, cx, cy, cw, ch, tint=t, z=3)
        ax.text(cx + cw / 2, cy + ch / 2, str(v), fontsize=fs, color=STRONG[t], ha="center",
                va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + cw / 2, y + 2 * ch + 3.0, "B: PASS", fontsize=8, color=FAINT, ha="center",
            fontfamily=SANS, zorder=5)
    ax.text(x + 1.5 * cw + 0.8, y + 2 * ch + 3.0, "B: FAIL", fontsize=8, color=FAINT, ha="center",
            fontfamily=SANS, zorder=5)
    ax.text(x - 1.2, y + 1.5 * ch + 0.8, "A: PASS", fontsize=8, color=FAINT, ha="right",
            va="center", fontfamily=SANS, zorder=5)
    ax.text(x - 1.2, y + ch / 2, "A: FAIL", fontsize=8, color=FAINT, ha="right",
            va="center", fontfamily=SANS, zorder=5)
    if title_txt:
        ax.text(x + cw + 0.4, y + 2 * ch + 7.2, title_txt, fontsize=9.4, color=NAVY, ha="center",
                fontweight="bold", fontfamily=SANS, zorder=5)


def kappa(a, b, c, d):
    n = a + b + c + d
    po = (a + d) / n
    pA, pB = (a + b) / n, (a + c) / n
    pe = pA * pB + (1 - pA) * (1 - pB)
    return po, pe, (po - pe) / (1 - pe)


# ══════════════════════════════════════════════ 01 big picture
fig, ax = canvas(13, 7.6); frame(ax)
header(ax, 1, "Every eval step is forced by a failure",
       "the lecture as a chain: each box exists because the one before it broke", glyph="map")
rows = [
    [("grey", "Try 5 examples", "selection bias, silent drift, no regression signal"),
     ("blue", "Golden dataset", "60 items: 50 MATH-500 + 10 stress tests"),
     ("blue", "Exact match", "26.7%  -  '0.5' vs '1/2' counted wrong"),
     ("blue", "Symbolic match", "fixes format for free, but can't judge reasoning")],
    [("teal", "LLM judge (Groq)", "43.3%  -  but lenient: passed missing answers"),
     ("teal", "2nd judge + kappa", "Gemini; 26/60 disagreements"),
     ("teal", "Escalate to humans", "agree -> automate, disagree -> review"),
     ("teal", "CI gate (DeepEval)", "assert_test fails the build; says THAT, not WHY")],
    [("teal", "Traces (Phoenix)", "max-steps 22, reasoning 12, parse error 10"),
     ("purple", "Judge = classifier", "measure TPR / TNR; correct the reported rate"),
     ("purple", "Error bars", "n = 60 -> 43.3% is really 31-56%"),
     ("purple", "Production loop", "sample live traffic; failures -> regression set")],
]
W, H, GX = 21.0, 15.5, 2.6
ys = [62.0, 40.0, 18.0]
for r, row in enumerate(rows):
    for c_, (t, head, txt) in enumerate(row):
        x = 3.5 + c_ * (W + GX)
        box(ax, x, ys[r], W, H, t, head, txt, fs=8.0, head_fs=9.4)
        if c_ < 3:
            arrow(ax, x + W + 0.3, ys[r] + H / 2, x + W + GX - 0.3, ys[r] + H / 2)
    if r < 2:
        xr = 3.5 + 3 * (W + GX) + W / 2
        ax.plot([xr, xr], [ys[r] - 0.3, ys[r] - 2.6], color=ARROW, lw=1.4, zorder=3)
        ax.plot([xr, 3.5 + W / 2], [ys[r] - 2.6, ys[r] - 2.6], color=ARROW, lw=1.4, zorder=3)
        arrow(ax, 3.5 + W / 2, ys[r] - 2.6, 3.5 + W / 2, ys[r + 1] + H + 0.3)
ax.text(96.5, 14.8, "purple = EXTENSION (beyond notebook)", fontsize=7.4, color=PURPLE, ha="right",
        fontweight="bold", fontfamily=MONO, zorder=5)
takeaway(ax, "You never pick an eval technique because it is fashionable. Each one exists because the "
             "previous step failed in a way you can measure.", y=1.5, h=10.5,
         right_lines=["26.7% -> 43.3%", "26 / 60 disagree", "+/- 12.5 pts"])
save(fig, "01_big_picture.png")


# ══════════════════════════════════════════════ 02 mental model
fig, ax = canvas(13, 6.6); frame(ax)
header(ax, 2, "The teacher and the head examiner",
       "a judge is a model you build and test - the loop at the bottom is what makes it trustworthy", glyph="model")
ins = [("grey", "Question / task"), ("grey", "Model's answer (+ trace)"),
       ("blue", "Rubric + expert critiques"), ("grey", "Gold answer (optional)")]
for i, (t, txt) in enumerate(ins):
    y = 66 - i * 11.0
    card(ax, 4, y, 22, 8.0, accent=t, z=3)
    ax.text(6.2, y + 4.0, txt, fontsize=8.8, color=NAVY if t == "blue" else BODY, va="center",
            fontweight="bold" if t == "blue" else "normal", fontfamily=SANS, zorder=5)
    arrow(ax, 26.4, y + 4.0, 35.6, 50.5)
card(ax, 36, 41, 20, 19, tint="indigo", accent="indigo", z=3)
ax.text(46, 54.5, "Judge LLM", fontsize=12, color=INDIGO, ha="center", va="center",
        fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(46, 48.5, "temperature 0", fontsize=8.4, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
ax.text(46, 44.8, "JSON out", fontsize=8.4, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
arrow(ax, 56.4, 50.5, 61.6, 50.5)
box(ax, 62, 41, 16, 19, "teal", "Output", "1. reasoning\n2. verdict", fs=8.6)
arrow(ax, 78.4, 50.5, 82.6, 50.5)
card(ax, 83, 52, 13, 7.5, tint="green", z=3); ax.text(89.5, 55.75, "PASS", fontsize=10, color=GREEN, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
card(ax, 83, 42, 13, 7.5, tint="rose", z=3); ax.text(89.5, 45.75, "FAIL", fontsize=10, color=ROSE, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
box(ax, 62, 17, 34, 13, "amber", "Head examiner: human labels",
    "expert re-marks a sample  ->  TPR, TNR, kappa, per slice", fs=8.4)
arrow(ax, 89.5, 41.6, 89.5, 30.4)
ax.plot([61.6, 15], [23.5, 23.5], color=AMBER, lw=1.4, ls="--", zorder=3)
arrow(ax, 15, 23.5, 15, 32.6, color=AMBER)
ax.text(38, 26.2, "disagreements -> fix the rubric and the examples", fontsize=8.6, color=AMBER,
        ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
takeaway(ax, "A teacher marks 10,000 essays with a marking scheme. You trust the marks only after a head "
             "examiner re-marks a sample and agrees - and you fix the scheme where they don't.", y=1.5, h=11.0)
save(fig, "02_mental_model.png")


# ══════════════════════════════════════════════ 03 eval pyramid
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 3, "The evaluation pyramid",
       "cheap and deterministic at the wide base; expensive and human at the narrow top", glyph="layers")
tiers = [  # bottom -> top
    ("blue",   "Exact / symbolic match", "free, instant, deterministic - run on everything", 70),
    ("blue",   "Custom rule metrics", "tool-usage score 0.96 - plain Python functions", 62),
    ("teal",   "Ragas (RAG metrics)", "faithfulness, context precision / recall", 54),
    ("teal",   "DeepEval GEval + pytest CI", "assert_test blocks the merge", 46),
    ("amber",  "Ensemble LLM judges + kappa", "Groq + Gemini; disagreements flagged", 38),
    ("rose",   "Human review", "validates the judges; the 'head examiner'", 30),
]
cx = 40.0
for i, (t, head, sub, w) in enumerate(tiers):
    y = 17.5 + i * 10.4
    card(ax, cx - w / 2, y, w, 8.6, tint=t, z=3)
    ax.text(cx, y + 5.7, head, fontsize=9.6, color=STRONG[t], ha="center", va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(cx, y + 2.6, sub, fontsize=7.6, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
arrow(ax, 77, 18.5, 77, 78.5, color=ROSE, lw=1.8)
ax.text(75.6, 66.0, "cost per item ↑\nlatency ↑\nsemantic power ↑", fontsize=8.2, color=ROSE,
        va="center", ha="right", fontfamily=UNI, zorder=5, fontweight="bold")
arrow(ax, 93, 78.5, 93, 18.5, color=GREEN, lw=1.8)
ax.text(91.6, 34.0, "coverage ↑\n(runs on\nmore items)", fontsize=8.0, color=GREEN, va="center", ha="right",
        fontfamily=UNI, zorder=5, fontweight="bold")
takeaway(ax, "Push every check as low in the pyramid as it can go. Use a judge only for what code cannot "
             "express, and humans only to keep the judges honest.", y=1.5, h=11.0,
         right_lines=["code first", "judge second", "humans on top"])
save(fig, "03_eval_pyramid.png")


# ══════════════════════════════════════════════ 04 scoreboard
fig, ax = canvas(13, 6.8); frame(ax)
header(ax, 4, "The class run: one agent, five accuracy numbers",
       "the same 60 predictions, measured five ways (recorded output, cells 50 / 58 / 70)", glyph="score")
bars = [("blue", "Exact match", 26.7, "16 / 60"), ("blue", "Symbolic match", 26.7, "16 / 60  (+0.0)"),
        ("teal", "Groq judge", 43.3, "26 / 60  (+16.7)"), ("rose", "Strict ensemble", 0.0, "both judges say correct"),
        ("amber", "Lenient ensemble", 43.3, "either judge says correct")]
for i, (t, lab, v, note) in enumerate(bars):
    y = 70 - i * 10.5
    hbar(ax, 4, y, 58, 4.6, v, t, lab, f"{v:.1f}%", lw=17, vw=9)
    ax.text(21, y - 2.2, note, fontsize=7.4, color=FAINT, va="center", fontfamily=MONO, zorder=5)
box(ax, 66, 49, 30.5, 26, "grey", "Why the agent scores low",
    "Stop reasons (cell 50):\nfinal answer  28\nmax steps     22\nparse error   10\n\nOnly 28 of 60 runs produced an answer at all.", fs=8.2)
strip(ax, 66, 32, 30.5, 13, "rose", "!",
      "Strict 0.0% is a pipeline bug, not a judgment: every Gemini call failed to parse.", fs=7.8)
strip(ax, 4, 15.5, 92.5, 8.0, "amber", "?",
      "Same predictions, 0% to 43% depending on the metric. '85% accuracy' means nothing without the metric, its kappa and its error bars.", fs=8.2)
takeaway(ax, "A score is a property of the measurement as much as of the model. Always report which "
             "ruler produced it.", y=1.5, h=10.5)
save(fig, "04_run_scoreboard.png")


# ══════════════════════════════════════════════ 05 two bugs
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 5, "Two silent bugs in the recorded run",
       "the notebook's text describes the pipeline; its output shows what actually happened", glyph="bug")
codeblock(ax, 3.5, 44, 45, 33, fs=8.4, step=1, step_tint="rose",
          title_text="Judge passed an answer that didn't exist  (cell 60)", code=(
"[math500_208]\n"
"  Predicted:  None\n"
"  Gold:       1\n"
"  Judge says: The student provided clear and\n"
"    correct reasoning, used the calculator\n"
"    appropriately, and arrived at the correct\n"
"    answer ...\n"
"\n"
"# also math500_3 and math500_178: None -> CORRECT"))
codeblock(ax, 51.5, 44, 45, 33, fs=8.4, step=2, step_tint="rose",
          title_text="Parse errors recorded as verdicts  (cell 68)", code=(
"[math500_210]\n"
"  Predicted: 10  |  Gold: 10\n"
"  Groq   -> CORRECT\n"
"  Gemini -> wrong: PARSE_ERROR: Expecting\n"
"            value: line 1 column 1\n"
"\n"
"# Gemini 'says wrong' on all 60 items ->\n"
"#   strict ensemble 0.0%, kappa = nan,\n"
"#   'observed agreement' still prints 57%"))
box(ax, 3.5, 16.5, 45, 22, "green", "Fix 1 - grade the outcome, not the trace",
    "Check answer presence with code BEFORE the judge runs. Add a hard rule: missing answer = FAIL. "
    "Keep 'answer correct' and 'reasoning quality' as separate judges.", fs=8.4)
box(ax, 51.5, 16.5, 45, 22, "green", "Fix 2 - three states, not two",
    "PASS / FAIL / ERROR. An ERROR is missing data: retry it or exclude it, never count it. "
    "Alert when the judge error rate goes above ~1%.", fs=8.4)
takeaway(ax, "Part of the +16.7-point 'judge gain' is leniency, and the 0% ensemble is broken plumbing. "
             "Neither shows up in the headline number.", y=1.5, h=11.0,
         right_lines=["None -> CORRECT", "ERROR -> FAIL", "both invisible"])
save(fig, "05_two_bugs.png")


# ══════════════════════════════════════════════ 06 kappa worked
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 6, "Cohen's kappa, worked on the class numbers",
       "exact match (rater A) vs the Groq judge (rater B) on 60 items", glyph="kappa")
matrix(ax, 14, 40, 13, 13, (14, 2, 12, 32), "rows = exact match, columns = Groq")
ax.text(4, 37.6, "Derived from printed margins:\nexact 16/60, Groq 26/60, 14 disagree\n-> a=14, b=2, c=12, d=32 (unique)",
        fontsize=8.0, color=BODY, va="top", fontfamily=SANS, zorder=5)
codeblock(ax, 46, 40, 50.5, 37, fs=8.8, step=1, step_tint="blue",
          title_text="The arithmetic", code=(
"po = (both PASS + both FAIL) / n\n"
"   = (14 + 32) / 60          = 0.767\n"
"\n"
"A says PASS 16/60 = 0.267\n"
"B says PASS 26/60 = 0.433\n"
"pe = 0.267*0.433 + 0.733*0.567 = 0.531\n"
"\n"
"kappa = (po - pe) / (1 - pe)\n"
"      = 0.236 / 0.469        = 0.502"))
ax.text(46, 33.5, "Scale (Landis & Koch):", fontsize=8.6, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
scale = [("rose", "<0.2", 0.0, 0.2), ("amber", "fair", 0.2, 0.4), ("amber", "moderate", 0.4, 0.6),
         ("teal", "substantial", 0.6, 0.8), ("green", "almost perfect", 0.8, 1.0)]
for t, lab, lo, hi in scale:
    x0 = 46 + lo * 50.5
    card(ax, x0, 24, 50.5 * (hi - lo) - 0.5, 6, tint=t, z=3)
    ax.text(x0 + 50.5 * (hi - lo) / 2, 27, lab, fontsize=7.6, color=STRONG[t], ha="center", va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
mx = 46 + 0.502 * 50.5
arrow(ax, mx, 37.2, mx, 30.6, color=NAVY, lw=1.8)
ax.text(mx + 1.0, 35.2, "0.502", fontsize=8.6, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(46 + 0.6 * 50.5, 21.0, "class trust bar: kappa >= 0.6", fontsize=7.6, color=TEAL, fontweight="bold",
        fontfamily=SANS, zorder=5)
strip(ax, 3.5, 15.5, 38, 11.5, "amber", "i",
      "77% raw agreement sounds good. Half of it (53%) is what chance alone would give.", fs=8.0)
takeaway(ax, "Kappa removes the agreement you would get by chance. Moderate (0.502) is below the 0.6 bar, "
             "so neither rater can stand in for the other.", y=1.5, h=11.0,
         right_lines=["po 0.767", "pe 0.531", "kappa 0.502"])
save(fig, "06_kappa_worked.png")


# ══════════════════════════════════════════════ 07 prevalence paradox
fig, ax = canvas(13, 6.6); frame(ax)
header(ax, 7, "The prevalence paradox: 90% agreement, kappa below zero",
       "when one label dominates, chance agreement is huge - percent agreement lies", glyph="trap")
ext_badge(ax)
for k, (vals, ttl, x) in enumerate([((54, 3, 3, 0), "Judges on PASS-heavy data", 13),
                                    ((40, 3, 2, 15), "A healthy pair of judges", 63)]):
    matrix(ax, x, 38, 11, 11, vals, ttl, fs=13)
    po, pe, kp = kappa(*vals)
    t = "rose" if kp < 0.6 else "green"
    box(ax, x - 9, 16.5, 42, 15, t, f"agreement {po*100:.0f}%   chance {pe*100:.1f}%   kappa {kp:+.2f}",
        ("They never agree on a single FAIL. Everything they 'agree' on is the majority label."
         if k == 0 else "They agree on most FAILs too - the cases you actually care about."), fs=8.2, head_fs=9.0)
takeaway(ax, "Report kappa AND the true-negative rate (how many real failures the judge catches). A judge "
             "that says PASS to everything scores high agreement on easy data.", y=1.5, h=11.0)
save(fig, "07_prevalence_paradox.png")


# ══════════════════════════════════════════════ 08 position bias
fig, ax = canvas(13, 6.4); frame(ax)
header(ax, 8, "Position bias, and why you judge both orders",
       "many pairs where A and B are EQUALLY good; the judge picks the first slot 70% of the time", glyph="swap")
ext_badge(ax)
def stacked(y, parts, lab):
    ax.text(4, y + 3.0, lab, fontsize=9.2, color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    x = 34
    for t, v, txt in parts:
        w = 60 * v
        if w > 0:
            ax.add_patch(K.FancyBboxPatch((x, y), w - 0.4, 6, boxstyle="round,pad=0,rounding_size=0.6",
                         fc=STRONG[t], ec="none", zorder=4))
            ax.text(x + w / 2, y + 3, txt, fontsize=8.6, color="white", ha="center", va="center",
                    fontweight="bold", fontfamily=SANS, zorder=5)
        x += w
p = 0.7
stacked(62, [("blue", p, "A wins 70%"), ("amber", 1 - p, "B 30%")], "One order (A first)")
stacked(48, [("blue", p * (1 - p), "A 21%"), ("amber", (1 - p) * p, "B 21%"),
             ("grey", p * p + (1 - p) ** 2, "tie 58%")], "Both orders, consistent wins")
codeblock(ax, 4, 17, 44, 21, fs=8.6, step=1, step_tint="blue", title_text="The math", code=(
"p = P(judge picks the first slot)\n"
"consistent A win = p * (1 - p)  = 0.21\n"
"consistent B win = (1 - p) * p  = 0.21\n"
"tie              = p^2 + (1-p)^2 = 0.58"))
box(ax, 52, 17, 44.5, 23, "green", "What this buys you",
    "The fake 70/30 preference becomes an honest tie. Arena-Hard judges every pair twice with "
    "positions swapped for exactly this reason. Cost: 2x judge calls.", fs=8.4)
takeaway(ax, "Run pairwise judgments in both orders. Count a win only when both agree; otherwise record "
             "a tie.", y=1.5, h=10.0)
save(fig, "08_position_bias.png")


# ══════════════════════════════════════════════ 09 build loop
fig, ax = canvas(13, 6.4); frame(ax)
header(ax, 9, "Building a judge you can trust",
       "critique shadowing (Husain & Shankar) - the class covered steps 4 and 6-7", glyph="loop")
ext_badge(ax)
steps = [("purple", "Error analysis", "read 20-100 real traces; note each failure"),
         ("purple", "One failure mode", "one judge per recurring failure"),
         ("purple", "Expert labels", "PASS/FAIL + written critique; split train/dev/test"),
         ("blue", "Judge prompt", "train split as few-shot examples"),
         ("purple", "TPR / TNR", "tune on dev, report once on test"),
         ("blue", "Deploy", "CI gate + live sample; correct the rate"),
         ("blue", "Re-validate", "new failures, judge or model changes")]
W = 12.0
for i, (t, head, txt) in enumerate(steps):
    x = 4.0 + i * (W + 1.3)
    card(ax, x, 40, W, 30, tint=t, z=3)
    stepbadge(ax, x + W / 2, 66, i + 1, t)
    ax.text(x + W / 2, 58.5, head, fontsize=8.6, color=STRONG[t], ha="center", va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    lines = tw.wrap(txt, 15)
    for j, ln in enumerate(lines):
        ax.text(x + W / 2, 52 - j * 3.2, ln, fontsize=7.4, color=BODY, ha="center", va="center",
                fontfamily=SANS, zorder=5)
    if i < 6:
        arrow(ax, x + W + 0.1, 55, x + W + 1.2, 55)
xl = 4.0 + 6 * (W + 1.3) + W / 2
ax.plot([xl, xl], [39.6, 33], color=AMBER, lw=1.4, ls="--", zorder=3)
ax.plot([xl, 4.0 + W / 2], [33, 33], color=AMBER, lw=1.4, ls="--", zorder=3)
arrow(ax, 4.0 + W / 2, 33, 4.0 + W / 2, 39.6, color=AMBER)
ax.text(50, 30.0, "criteria drift: grading teaches you new failure modes -> new judges, new regression cases",
        fontsize=8.4, color=AMBER, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(50, 24.5, "blue = covered in class    purple = extension", fontsize=7.8, color=FAINT, ha="center",
        fontfamily=SANS, zorder=5)
takeaway(ax, "You don't write a judge and ship it. You train it like a model: labelled data, a held-out "
             "test split, and a measured error rate.", y=1.5, h=11.0,
         right_lines=["100-200 labels", "~half FAILs", "test once"])
save(fig, "09_build_loop.png")


# ══════════════════════════════════════════════ 10 judge correction
fig, ax = canvas(13, 6.8); frame(ax)
header(ax, 10, "A judge is a classifier - correct its number",
       "the judge's pass rate mixes the truth with the judge's own mistakes", glyph="fix")
ext_badge(ax)
box(ax, 4, 50, 28, 26, "green", "TPR  (sensitivity)",
    "Of the truly GOOD outputs, how many does the judge PASS?\nLow TPR = too harsh.", fs=8.4)
box(ax, 4, 20, 28, 26, "rose", "TNR  (specificity)",
    "Of the truly BAD outputs, how many does the judge FAIL?\nLow TNR = misses failures, inflates the score.", fs=8.4)
codeblock(ax, 36, 40, 60.5, 36, fs=8.8, step=1, step_tint="teal",
          title_text="Rogan-Gladen correction (used by the evals course's judgy tool)", code=(
"theta = (p_obs + TNR - 1) / (TPR + TNR - 1)\n"
"\n"
"# class judge reported p_obs = 0.433\n"
"# ILLUSTRATIVE judge quality: TPR 0.95, TNR 0.80\n"
"theta = (0.433 + 0.80 - 1) / (0.95 + 0.80 - 1)\n"
"      = 0.233 / 0.75\n"
"      = 0.311          # 31.1%, not 43.3%\n"
"\n"
"# only valid if TPR + TNR > 1 (judge beats chance)"))
strip(ax, 36, 20, 60.5, 14, "amber", "i",
      "The class never measured TPR/TNR, so the true rate is unknown. That is the point: without "
      "labelled data you cannot tell 43% from 31%.", fs=8.2)
takeaway(ax, "Measure the judge on 100-200 expert labels, then report the corrected rate with a bootstrap "
             "confidence interval - not the raw judge score.", y=1.5, h=11.0)
save(fig, "10_judge_correction.png")


# ══════════════════════════════════════════════ 11 error bars
fig, ax = canvas(13, 6.6); frame(ax)
header(ax, 11, "How sure is '43.3%'? Error bars on 60 items",
       "an eval score is an estimate from a sample - it needs a confidence interval", glyph="stats")
ext_badge(ax)
def ci(p, n):
    se = math.sqrt(p * (1 - p) / n)
    return p - 1.96 * se, p + 1.96 * se
X0, XW = 30, 62
for k in range(0, 101, 10):
    x = X0 + XW * k / 100
    ax.plot([x, x], [44, 74], color="#E3E8F0", lw=0.8, zorder=1)
    ax.text(x, 41.5, f"{k}%", fontsize=7.4, color=FAINT, ha="center", fontfamily=SANS, zorder=5)
for i, (t, lab, p) in enumerate([("blue", "Exact match", 16 / 60), ("teal", "Groq judge", 26 / 60)]):
    y = 64 - i * 13
    lo, hi = ci(p, 60)
    ax.text(4, y, lab, fontsize=9.4, color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(4, y - 3.4, f"{p*100:.1f}%  [{lo*100:.1f} - {hi*100:.1f}]", fontsize=7.8, color=FAINT,
            va="center", fontfamily=MONO, zorder=5)
    ax.plot([X0 + XW * lo, X0 + XW * hi], [y, y], color=STRONG[t], lw=6, solid_capstyle="round", zorder=4)
    ax.plot([X0 + XW * p], [y], "o", ms=9, color="white", mec=STRONG[t], mew=2.4, zorder=5)
codeblock(ax, 4, 13.5, 44, 18.5, fs=8.6, step=1, step_tint="teal", title_text="Worked", code=(
"SE = sqrt(p(1-p)/n) = sqrt(0.433*0.567/60)\n"
"   = 0.064\n"
"95% CI = 0.433 +/- 1.96*0.064\n"
"       = 30.8% ... 55.9%"))
codeblock(ax, 52, 13.5, 44.5, 18.5, fs=8.6, step=2, step_tint="teal", title_text="Items needed for +/-3 points", code=(
"n = (1.96 * 0.5 / 0.03)^2\n"
"  = 32.7^2\n"
"  = 1,067 items\n"
"# Jason Wei: aim for >= 1,000"))
takeaway(ax, "With 60 items you can only see huge changes. Use paired tests on the same items, enough n, "
             "and clustered errors when items share a source.", y=1.5, h=9.5)
save(fig, "11_error_bars.png")


# ══════════════════════════════════════════════ 12 agent views
fig, ax = canvas(13, 7.0); frame(ax)
header(ax, 12, "Evaluating an agent: outcome, trajectory, reliability",
       "a ReAct agent can be right the wrong way, or wrong in a way only the trace shows", glyph="agent")
box(ax, 4, 52, 29, 24, "blue", "Outcome",
    "Is the end state right?\nClass: exact / symbolic match.\nGrade the result, not what the agent says it did.", fs=8.2)
box(ax, 35.5, 52, 29, 24, "teal", "Trajectory",
    "Were the steps sensible?\nClass: tool-usage score 0.96 (55 good, 5 with errors); failure categories from traces.", fs=8.2)
box(ax, 67, 52, 29.5, 24, "purple", "Reliability",
    "Does it hold up?\nClass: self-consistency at n=4.\nExtension: pass^k = all k tries succeed.", fs=8.2)
ax.text(4, 46.5, "Why the 44 exact-match failures happened  (traces, cell 89)", fontsize=9.2, color=NAVY,
        fontweight="bold", fontfamily=SANS, zorder=5)
for i, (lab, v, t) in enumerate([("hit_max_steps", 22, "rose"), ("wrong_reasoning", 12, "amber"),
                                 ("parse_error", 10, "amber")]):
    hbar(ax, 4, 38 - i * 7.4, 58, 4.4, v / 44 * 100, t, lab, str(v), lw=18, vw=6)
codeblock(ax, 66, 17, 30.5, 26, fs=8.6, step=1, step_tint="purple", title_text="pass@k vs pass^k", code=(
"per-try success p = 0.7\n"
"pass@3 = 1-(1-p)^3\n"
"       = 0.973\n"
"pass^3 = p^3\n"
"       = 0.343\n"
"# users get ONE try"))
strip(ax, 4, 13.6, 58, 6.6, "teal", "i", "Metrics say WHETHER it fails. Traces say WHY - and what to fix first.", fs=8.2)
takeaway(ax, "Score the outcome with code, the path with checks plus a judge, and reliability with repeated "
             "tries. One accuracy number hides all three.", y=1.5, h=10.5)
save(fig, "12_agent_views.png")


# ══════════════════════════════════════════════ 13 production architecture
fig, ax = canvas(13, 7.6); frame(ax)
header(ax, 13, "Production eval architecture",
       "the class built the offline half; production adds an online half", glyph="prod")
ext_badge(ax)
ax.text(4, 77.5, "ONLINE - REQUEST PATH  (budget: milliseconds)", fontsize=7.8, color=FAINT, fontweight="bold", fontfamily=MONO, zorder=5)
for x, w, t, h1, h2 in [(4, 12, "grey", "User", ""), (20, 19, "blue", "Agent / app", "ReAct + tools"),
                        (43, 27, "amber", "Inline guardrails", "schema, PII, policy - ms, deterministic"),
                        (74, 15, "grey", "Response", "")]:
    card(ax, x, 64, w, 10, tint=t, z=3)
    ax.text(x + w / 2, 70.5 if h2 else 69, h1, fontsize=9, color=STRONG[t] if t != "grey" else NAVY, ha="center",
            va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    if h2:
        ax.text(x + w / 2, 66.8, h2, fontsize=7.0, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
for x1, x2 in [(16.2, 19.8), (39.2, 42.8), (70.2, 73.8)]:
    arrow(ax, x1, 69, x2, 69)
ax.text(43, 57.5, "ONLINE - ASYNC  (never blocks the user)", fontsize=7.8, color=FAINT, fontweight="bold", fontfamily=MONO, zorder=5)
for x, w, t, h1, h2 in [(20, 19, "grey", "Trace store", "OTel - Phoenix / Langfuse"),
                        (43, 22, "grey", "Stratified sampler", "1-2% + escalations, thumbs-down"),
                        (69, 27.5, "teal", "Judge service", "versioned; cascade small -> large")]:
    card(ax, x, 44, w, 10, tint=t if t != "grey" else None, accent=t, z=3)
    ax.text(x + w / 2, 50.6, h1, fontsize=9, color=NAVY if t == "grey" else STRONG[t], ha="center", va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + w / 2, 46.8, h2, fontsize=7.0, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
arrow(ax, 29.5, 63.6, 29.5, 54.4)
arrow(ax, 39.2, 49, 42.8, 49); arrow(ax, 65.2, 49, 68.8, 49)
card(ax, 69, 30, 12.5, 10, tint="indigo", z=3)
ax.text(75.25, 36.3, "Dashboards", fontsize=8.6, color=INDIGO, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(75.25, 32.6, "rate +/- CI, alerts", fontsize=6.9, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
card(ax, 84, 30, 12.5, 10, tint="rose", z=3)
ax.text(90.25, 36.3, "Human queue", fontsize=8.6, color=ROSE, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(90.25, 32.6, "disagree / unsure", fontsize=6.9, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
arrow(ax, 75.25, 43.6, 75.25, 40.4); arrow(ax, 90.25, 43.6, 90.25, 40.4)
ax.text(4, 37.5, "OFFLINE - PRE-DEPLOY", fontsize=7.8, color=FAINT, fontweight="bold", fontfamily=MONO, zorder=5)
card(ax, 4, 19, 30, 13, tint="blue", z=3)
ax.text(19, 27.6, "CI gate on every PR", fontsize=9, color=BLUE, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(19, 23.0, "frozen set, code checks + judges", fontsize=7.0, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
card(ax, 40, 19, 38, 9, tint="purple", z=3)
ax.text(59, 24.9, "Labelled store", fontsize=9, color=PURPLE, ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(59, 21.3, "regression set + judge TPR/TNR control set", fontsize=7.0, color=BODY, ha="center", va="center", fontfamily=SANS, zorder=5)
ax.plot([90.25, 90.25], [29.6, 23.5], color=ARROW, lw=1.4, zorder=3)
arrow(ax, 90.25, 23.5, 78.4, 23.5)
arrow(ax, 39.6, 25.5, 34.4, 25.5)
ax.plot([10, 10], [32.4, 35.4], color=ARROW, lw=1.2, ls="--", zorder=3)
ax.plot([10, 17.6], [35.4, 35.4], color=ARROW, lw=1.2, ls="--", zorder=3)
ax.plot([17.6, 17.6], [35.4, 66], color=ARROW, lw=1.2, ls="--", zorder=3)
arrow(ax, 17.6, 66, 19.8, 66)
ax.text(18.6, 39.0, "gates deploys of the agent", fontsize=7.0, color=FAINT, fontfamily=SANS, zorder=5)
takeaway(ax, "Guardrails sit in the request path, so they must be milliseconds and deterministic. Judges run "
             "async on a sample. Humans produce the labels that keep both honest.", y=1.5, h=11.0)
save(fig, "13_production_architecture.png")


# ══════════════════════════════════════════════ 14 decision tree
fig, ax = canvas(13, 7.4); frame(ax)
header(ax, 14, "Which evaluation method?",
       "start at the top; stop at the first YES", glyph="choose")
qs = [("Can a program check it?", "exact number, valid JSON, tests pass, regex, length",
       "Deterministic check - no LLM", "symbolic_match, schema, unit tests. Free, instant, unbiased."),
      ("Do you have a gold answer per case?", "math, factual Q&A with varied formats",
       "Reference-guided judge, binary", "'Is it equivalent to GOLD?' + reasoning first. Validate TPR/TNR."),
      ("Comparing two versions?", "prompt A vs B, model X vs Y",
       "Pairwise judge, both orders", "Consistent wins only; win rate +/- CI; control for length."),
      ("Is it a multi-step agent?", "tool calls, loops, state",
       "Trajectory checks + outcome judge", "Code for tool calls and loops; traces for why.")]
for i, (q, sub, ans, why) in enumerate(qs):
    y = 69 - i * 13.0
    card(ax, 4, y, 40, 10.5, accent="blue", z=3)
    ax.text(6.4, y + 7.0, q, fontsize=9.4, color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(6.4, y + 3.2, sub, fontsize=7.6, color=FAINT, va="center", fontfamily=SANS, zorder=5)
    arrow(ax, 44.4, y + 5.25, 50.6, y + 5.25, color=GREEN)
    ax.text(47.5, y + 7.3, "yes", fontsize=7.6, color=GREEN, ha="center", fontweight="bold", fontfamily=SANS, zorder=5)
    card(ax, 51, y, 45.5, 10.5, tint="green", z=3)
    ax.text(53.4, y + 7.0, ans, fontsize=9.2, color=GREEN, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(53.4, y + 3.2, why, fontsize=7.6, color=BODY, va="center", fontfamily=SANS, zorder=5)
    if i < 3:
        arrow(ax, 12, y - 0.3, 12, y - 2.2, color=ROSE)
        ax.text(13.2, y - 1.3, "no", fontsize=7.4, color=ROSE, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
arrow(ax, 12, 29.7, 12, 27.2, color=ROSE)
ax.text(13.2, 28.4, "no", fontsize=7.4, color=ROSE, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
card(ax, 4, 16, 92.5, 10.8, tint="teal", z=3)
ax.text(6.4, 23.3, "Pointwise rubric judge: one criterion, PASS / FAIL", fontsize=9.4, color=TEAL, va="center",
        fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(6.4, 19.4, "One judge per failure mode from error analysis; few-shot expert critiques; reasoning first; temperature 0; validate, then correct the rate.",
        fontsize=7.6, color=BODY, va="center", fontfamily=SANS, zorder=5)
takeaway(ax, "Never present one technique as the default. The question you can answer with the cheapest "
             "reliable tool decides the method.", y=1.5, h=10.5)
save(fig, "14_decision_tree.png")
print("done")
