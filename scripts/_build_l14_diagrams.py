"""Diagrams for L14 - DSPy: Mathematical Prompt Optimization.

Built from scripts/diagram_kit.py. All code shown is real notebook code; all
numbers are the ones verified by scripts/verify_dspy_results.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_kit import *   # noqa
import diagram_kit as K
import textwrap as tw

set_output(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "8. Agentic AI Systems", "14. DSPy: Mathematical Prompt Optimization", "images")))


def strip(ax, x, y, w, h, tint, glyph, text, fs=7.9):
    card(ax, x, y, w, h, tint=tint, accent=tint, z=3)
    chip(ax, x + 4.2, y + h / 2, tint, glyph, r=1.8, fs=8.5)
    lines = tw.wrap(text, wrap_chars(ax, w - 12.0, fs))
    lh = K._lineh(ax, fs, 1.7)
    yy = y + h / 2 + lh * (len(lines) - 1) / 2
    for ln in lines:
        ax.text(x + 7.6, yy, ln, fontsize=fs, color=STRONG[tint], va="center",
                fontfamily=SANS, zorder=5, fontweight="bold")
        yy -= lh


def scorerow(ax, x, y, w, h, label, sub, pct, tint, note="", lw=30, big=13.0, vg=19.0):
    """Label + proportional bar + value. The bar is dropped when the card is too
    narrow to hold one -- a zero/negative bar width renders as a misleading dot."""
    card(ax, x, y, w, h, z=3, accent=tint)
    ax.text(x + 3.4, y + h * 0.62, label, fontsize=8.8, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + 3.4, y + h * 0.26, sub, fontsize=7.0, color=FAINT, va="center",
            fontfamily=MONO, zorder=5)
    bx, bw = x + lw, w - lw - vg
    if bw >= 8.0:
        ax.add_patch(K.FancyBboxPatch((bx, y + h / 2 - 1.3), bw, 2.6,
                     boxstyle="round,pad=0,rounding_size=1.2", fc="#EDF1F7", ec="none", zorder=4))
        ax.add_patch(K.FancyBboxPatch((bx, y + h / 2 - 1.3), max(bw * pct / 100.0, 1.2), 2.6,
                     boxstyle="round,pad=0,rounding_size=1.2", fc=STRONG[tint], ec="none", zorder=5))
        if note:
            ax.text(bx, y + h * 0.18, note, fontsize=6.9, color=BODY, va="center",
                    fontfamily=SANS, zorder=6)
    elif note:
        ax.text(x + 3.4, y + h * 0.10, note, fontsize=6.9, color=BODY, va="center",
                fontfamily=SANS, zorder=6)
    ax.text(x + w - 3.0, y + h / 2, f"{pct:.2f}%", fontsize=big, color=STRONG[tint],
            va="center", ha="right", fontweight="bold", fontfamily=SANS, zorder=6)


# ══════════════════════════════════════════════ 01
fig, ax = canvas(13, 7.0); frame(ax)
header(ax, 1, "You write the contract. DSPy writes the prompt.",
       "five lines of Python on the left produced every line on the right", glyph="dspy")
codeblock(ax, 3.5, 24, 45.5, 52, fs=9.2, step=1, step_tint="green",
          title_text="What you write  (cell 22)", code=(
"class FinancialSentimentBasic(dspy.Signature):\n"
'    """Classify the sentiment of a\n'
'    financial news headline."""\n'
"\n"
"    headline: str = dspy.InputField()\n"
"    sentiment: str = dspy.OutputField(\n"
"        desc=\"one of: positive,\n"
"              negative, neutral\")\n"
"\n"
"basic = dspy.Predict(FinancialSentimentBasic)\n"
"basic(headline=sample)\n"
"\n"
"# -> 'negative'\n"
"#\n"
"# You never wrote a prompt.\n"
"# You declared a typed contract."))
codeblock(ax, 51.5, 24, 45.0, 52, fs=9.2, step=2, step_tint="blue",
          title_text="What DSPy sends  (cell 24, inspect_history)", code=(
"Your input fields are:\n"
"1. `headline` (str):\n"
"Your output fields are:\n"
"1. `sentiment` (str): one of: positive,\n"
"   negative, neutral\n"
"All interactions will be structured in the\n"
"following way, with the appropriate values\n"
"filled in.\n"
"\n"
"[[ ## headline ## ]]\n"
"{headline}\n"
"\n"
"[[ ## sentiment ## ]]\n"
"{sentiment}\n"
"\n"
"[[ ## completed ## ]]\n"
"In adhering to this structure, your objective\n"
"is: Classify the sentiment of a financial\n"
"news headline."))
strip(ax, 3.5, 15.5, 93.0, 6.4, "amber", "i",
      "The docstring became the objective. The field names became the [[ ## ## ]] slots. The desc= "
      "became the constraint. Nothing here was hand-tuned - it is compiled output.")
takeaway(ax, "A prompt is build output, not source code. The source is a typed signature; the "
             "prompt is what the adapter generates from it.",
         right_lines=["5 lines in", "19 lines out", "zero prompt edits"])
save(fig, "01_contract_not_prompt.png")


# ══════════════════════════════════════════════ 02
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 2, "Why hand-tuned prompts rot",
       "the notebook's 3 AM debugging story, as a causal chain", glyph="why")
rows = [
    ("1", "grey",  "Problem",
     "Classify financial headlines the way an equity investor would. 'Fed signals patience on rate "
     "cuts' is neutral English and negative for growth stocks."),
    ("2", "blue",  "Naive approach",
     "Write a prompt. Add 'think step by step' (58% -> 63%). Paste in three examples (-> 71%)."),
    ("3", "rose",  "Limitation",
     "Thursday 3 AM you add a format constraint and it drops to 68%. Friday the CTO swaps the model "
     "and it falls to 54%. Every fix broke something else."),
    ("4", "green", "Solution",
     "Declare inputs, outputs and a metric. Let a search procedure choose the wording and the "
     "demonstrations against a measured score."),
    ("5", "amber", "Trade-off",
     "You now need labelled data, a metric you trust, and an API budget - and as this notebook "
     "shows, the search can still lose to the baseline."),
]
y = 68
for n, tint, head, txt in rows:
    card(ax, 4, y, 93, 10.4, z=3, accent=tint)
    stepbadge(ax, 9.0, y + 5.2, n, tint, r=2.0, fs=9.5)
    ax.text(13.5, y + 5.2, head, fontsize=10.0, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    body(ax, 31.0, y + 6.7, txt, width=84, fs=8.5, lh=2.7)
    y -= 11.6
takeaway(ax, "DSPy does not write nicer English. It makes the prompt measurable, so an algorithm "
             "can search it - and so you can tell when the search failed.",
         y=3.0, h=11.5, right_lines=["prompt -> parameter", "eyeball -> metric", "edit -> compile"])
save(fig, "02_why_prompts_rot.png")


# ══════════════════════════════════════════════ 03
fig, ax = canvas(13, 8.0); frame(ax)
header(ax, 3, "The three pillars, plus the one that decides everything",
       "Signature declares - Module strategises - Optimizer searches - Metric judges", glyph="core")
cards = [
    (3.0,  "blue",   "S", "Signature", [
        "Typed I/O contract.", "Docstring = the instruction.",
        "desc= = per-field constraint.", "Says WHAT, never HOW."]),
    (27.2, "purple", "M", "Module", [
        "The execution strategy.", "Predict = one shot.",
        "ChainOfThought = + reasoning field.",
        "ReAct / ProgramOfThought also available."]),
    (51.4, "amber",  "O", "Optimizer", [
        "The compiler. a.k.a. teleprompter.",
        "BootstrapFewShot picks demos.",
        "MIPROv2 also rewrites the instruction.",
        "compile() returns a NEW module."]),
    (75.6, "teal",   "f", "Metric", [
        "(example, pred, trace) -> float.",
        "The only definition of 'better'.",
        "Optimizer inherits all its blind spots.",
        "Not a pillar in the docs - but it rules them."]),
]
for x, tint, g, head, lines in cards:
    railcard(ax, x, 45, 21.8, 30, tint, g, head, lines, fs_head=9.8, fs_body=7.5)
codeblock(ax, 3.0, 17, 94.4, 23, fs=9.6, step=5, step_tint="indigo",
          title_text="How they compose  (cells 32, 44, 50)", code=(
"def sentiment_match(example, pred, trace=None):          # METRIC\n"
"    return float(example.sentiment.strip().lower() == predicted)\n"
"\n"
"student   = dspy.ChainOfThought(FinancialSentiment)      # MODULE wrapping a SIGNATURE\n"
"bootstrap = BootstrapFewShot(metric=sentiment_match, max_bootstrapped_demos=4)\n"
"mipro     = MIPROv2(metric=sentiment_match, auto=\"light\", num_threads=4)   # OPTIMIZERS"))
takeaway(ax, "Signature = what. Module = how. Optimizer = search. Metric = better. "
             "Change any one without rewriting the other three.",
         y=2.0, h=13.0, right_lines=["declare", "strategise", "search", "judge"])
save(fig, "03_three_pillars.png")


# ══════════════════════════════════════════════ 04
fig, ax = canvas(13, 7.6); frame(ax)
header(ax, 4, "The result the notebook did not expect",
       "four programs, one 102-example test set, every optimizer BELOW the baseline", glyph="!! ")
ax.text(3.5, 79.0, "Method", fontsize=8.2, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(94.0, 79.0, "Test accuracy", fontsize=8.2, color=FAINT, fontweight="bold",
        ha="right", fontfamily=SANS, zorder=5)
scorerow(ax, 3.5, 65.5, 93.0, 11.0, "Predict  (baseline, unoptimized)", "80 / 102", 78.43,
         "green", "the best result in the notebook - and nobody optimized it")
scorerow(ax, 3.5, 53.5, 93.0, 11.0, "BootstrapFewShot  (4 demos)", "77 / 102", 75.49,
         "amber", "best optimizer, still 2.94 pts below doing nothing")
scorerow(ax, 3.5, 41.5, 93.0, 11.0, "ChainOfThought  (baseline)", "74 / 102", 72.55,
         "rose", "reasoning made it worse on one-sentence headlines")
scorerow(ax, 3.5, 29.5, 93.0, 11.0, "MIPROv2  (auto='light')", "74 / 102", 72.55,
         "rose", "identical to CoT - see diagram 8 for why")
ax.plot([3.5 + 30 + (93.0-30-19.0) * 0.80] * 2, [29.0, 77.0], color=ROSE, lw=1.4,
        ls="--", zorder=7)
ax.text(3.5 + 30 + (93.0-30-19.0) * 0.80 + 0.8, 78.0, "the chart's 80% target", fontsize=7.2,
        color=ROSE, va="center", fontfamily=SANS, zorder=7)
strip(ax, 3.5, 17.5, 93.0, 8.5, "rose", "!",
      "The wrap-up claims 'you just moved a real classifier from ~60% to 80%+ accuracy'. No run "
      "reached 80%, no run started at 60%, and the three optimized runs all finished below the "
      "unoptimized one. The dashed line is the notebook's own target, drawn in its own chart.")
takeaway(ax, "A negative result, honestly recorded. This is the most useful thing in the lecture - "
             "optimization is a hypothesis you test, not a step that always pays.",
         y=2.5, h=12, right_lines=["baseline 78.43%", "best optimizer 75.49%", "nothing hit 80%"])
save(fig, "04_the_result.png")


# ══════════════════════════════════════════════ 05
fig, ax = canvas(13, 7.0); frame(ax)
header(ax, 5, "Predict vs ChainOfThought",
       "same signature, one extra output field - and it cost 5.88 points", glyph="module")
codeblock(ax, 3.5, 44, 44.0, 30, fs=9.4, step=1, step_tint="blue",
          title_text="dspy.Predict  -> the adapter asks for 1 field", code=(
"Your output fields are:\n"
"1. `sentiment` (str): one of: positive,\n"
"   negative, neutral\n"
"\n"
"[[ ## sentiment ## ]]\n"
"{sentiment}\n"
"\n"
"Respond ... starting with the field\n"
"`[[ ## sentiment ## ]]`"))
codeblock(ax, 53.0, 44, 43.5, 30, fs=9.4, step=2, step_tint="purple",
          title_text="dspy.ChainOfThought -> inserts `reasoning` FIRST", code=(
"Your output fields are:\n"
"1. `reasoning` (str):\n"
"2. `sentiment` (str): one of: positive,\n"
"   negative, neutral\n"
"\n"
"[[ ## reasoning ## ]]\n"
"{reasoning}\n"
"\n"
"[[ ## sentiment ## ]]\n"
"{sentiment}"))
scorerow(ax, 3.5, 30.0, 45.0, 10.0, "Predict", "80 / 102   -   10.8s", 78.43, "green", "", lw=17, vg=24, big=12.5)
scorerow(ax, 52.0, 30.0, 44.5, 10.0, "ChainOfThought", "74 / 102   -   7.3s", 72.55, "rose", "", lw=20, vg=24, big=12.5)
strip(ax, 3.5, 17.5, 93.0, 9.0, "amber", "i",
      "Why reasoning HURTS here: the labels are annotator consensus on one-sentence corporate "
      "announcements. Told to argue first, the model finds an argument - and a story about a "
      "company is almost always optimistic. The gold label is just 'an announcement happened'.")
takeaway(ax, "Chain of Thought buys deliberation. On multi-step problems that is a bargain; on "
             "three-way classification of short headlines it is a 5.88-point loss.",
         y=2.5, h=12, right_lines=["+1 output field", "-5.88 points", "measure, don't assume"])
save(fig, "05_predict_vs_cot.png")


# ══════════════════════════════════════════════ 06
fig, ax = canvas(13, 7.0); frame(ax)
header(ax, 6, "Splitting data for an optimizer, not a model",
       "3,872 headlines -> 21 / 51 / 102, and each split has a different job", glyph="data")
ax.text(3.5, 76.5, "Raw Financial PhraseBank  (cell 16)  -  heavily imbalanced",
        fontsize=8.8, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
x = 3.5
for name, n, tint in [("neutral", 2298, "grey"), ("positive", 1091, "blue"), ("negative", 483, "rose")]:
    w = 93.0 * n / 3872
    card(ax, x, 66.5, w - 0.5, 7.2, tint=tint, accent=tint, z=3)
    ax.text(x + (w - 0.5) / 2, 71.4, name, fontsize=8.0, color=STRONG[tint], ha="center",
            va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + (w - 0.5) / 2, 68.4, f"{n}   {100*n/3872:.1f}%", fontsize=7.4,
            color=STRONG[tint], ha="center", va="center", fontfamily=MONO, zorder=5)
    x += w
strip(ax, 3.5, 57.5, 93.0, 6.6, "rose", "!",
      "Always answering 'neutral' scores 2298/3872 = 59.35% on this pool. The balanced 34/34/34 "
      "test set drops the trivial floor to 33.33% - that is the number 78.43% must be read against.")
sp = [(3.5,  "blue",  "trainset", "21", "7 per class", "7 pos / 7 neg / 7 neu",
       "Source of demonstrations. The optimizer runs the teacher over these and keeps traces that score 1.0."),
      (35.7, "purple","valset", "51", "17 per class", "17 pos / 17 neg / 17 neu",
       "Scores candidate programs during search. BootstrapFewShot ignores it; MIPROv2 actually uses it."),
      (67.9, "teal",  "testset", "102", "34 per class", "34 pos / 34 neg / 34 neu",
       "Final measurement. All four evaluations run against this same set.")]
for x0, tint, nm, n, per, cls, desc in sp:
    card(ax, x0, 28.0, 28.6, 25.5, z=3)
    chip(ax, x0 + 4.4, 49.2, tint, nm[0].upper(), r=2.1, fs=9)
    ax.text(x0 + 8.4, 49.8, nm, fontsize=9.6, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(x0 + 8.4, 46.6, per, fontsize=7.3, color=FAINT, va="center", fontfamily=SANS, zorder=5)
    ax.text(x0 + 25.0, 48.4, n, fontsize=17, color=STRONG[tint], va="center", ha="right",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x0 + 3.2, 42.0, cls, fontsize=7.6, color=STRONG[tint], va="center",
            fontfamily=MONO, zorder=5)
    body(ax, x0 + 3.2, 37.5, desc, width=40, fs=7.6, lh=2.7)
strip(ax, 3.5, 19.0, 93.0, 6.2, "amber", "i",
      "58 of the 483 negatives (12.0%) are consumed by the three splits - the scarce class caps how "
      "large a balanced split can get. 21 training examples is very little to search against.")
takeaway(ax, "Stratify, or your metric measures the class prior instead of the program. "
             "Train is a demo pool here - not gradient data.",
         right_lines=["train = demos", "val = search", "test = once"])
save(fig, "06_data_splits.png")

# ══════════════════════════════════════════════ 07
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 7, "BootstrapFewShot, step by step",
       "the model writes its own few-shot examples - and the metric decides which survive",
       glyph="compile")
steps = [
    (3.0,  "blue",   "1", "Teacher runs",
     "An unoptimized copy of the module runs on a training example and emits a full trace: reasoning + answer."),
    (27.2, "teal",   "2", "Metric filters",
     "sentiment_match scores the trace against the gold label. Score < 1.0 and the trace is discarded."),
    (51.4, "green",  "3", "Keep k traces",
     "Survivors become demonstrations. Capped by max_bootstrapped_demos=4."),
    (75.6, "purple", "4", "Student compiled",
     "Demos are baked into the student's prompt. compile() returns a new module; the original is untouched."),
]
for x, tint, n, head, txt in steps:
    card(ax, x, 55, 21.8, 21, z=3)
    stepbadge(ax, x + 4.0, 72.2, n, tint, r=2.0, fs=9.5)
    ax.text(x + 8.0, 72.2, head, fontsize=9.2, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    body(ax, x + 3.2, 66.4, txt, width=33, fs=7.6, lh=2.75)
    if x < 75:
        arrow(ax, x + 22.2, 65.5, x + 26.8, 65.5, color=STRONG[tint], lw=1.9, ms=11)
codeblock(ax, 3.0, 20, 55.0, 27, fs=9.0, step=5, step_tint="amber",
          title_text="cell 44 - and what it actually reported", code=(
"bootstrap = BootstrapFewShot(\n"
"    metric=sentiment_match,\n"
"    max_bootstrapped_demos=4,\n"
"    max_labeled_demos=4,\n"
"    max_rounds=1)\n"
"\n"
"# Bootstrapped 4 full traces after 4 examples\n"
"# Compile time: 0.2s"))
railcard(ax, 61.0, 20, 35.5, 27, "rose", "!", "Read that output again", [
    "It stopped after 4 of 21 examples - 19%.",
    "The first 4 all scored 1.0, hitting the cap.",
    "17 training examples were never seen.",
    "Your demos are the first 4 that happened to work, not the best 4."], fs_body=7.6)
takeaway(ax, "BootstrapFewShot is a filter that stops at the cap, not a search. It scored 75.49% - "
             "the best of the three optimizers, and still 2.94 points below doing nothing.",
         right_lines=["4 of 21 used", "0.2s compile", "first-fit, not best-fit"])
save(fig, "07_bootstrap_algorithm.png")


# ══════════════════════════════════════════════ 08
fig, ax = canvas(13, 7.6); frame(ax)
header(ax, 8, "Why MIPROv2 scored EXACTLY the same as the baseline",
       "not a coincidence - it returned the program it started from", glyph="audit")
ax.text(3.5, 78.0, "MIPROv2 is supposed to optimise BOTH halves of the prompt:",
        fontsize=8.8, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
for x, tint, t, d in [(3.5, "purple", "instruction", "proposed by the LLM, scored on the valset"),
                      (50.0, "blue", "demonstrations", "combinations searched with a TPE surrogate")]:
    card(ax, x, 66.0, 46.5, 9.0, z=3, accent=tint)
    chip(ax, x + 4.2, 70.5, tint, t[0].upper(), r=1.9, fs=9)
    ax.text(x + 8.0, 71.8, t, fontsize=9.0, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(x + 8.0, 68.6, d, fontsize=7.3, color=BODY, va="center", fontfamily=SANS, zorder=5)
codeblock(ax, 3.5, 27.5, 55.0, 30, fs=8.6, step=1, step_tint="rose",
          title_text="cell 51 - what it actually returned", code=(
"print(compiled_mipro.predict.signature.instructions)\n"
"\n"
"# Classify the sentiment of a financial news\n"
"# headline from the perspective of an equity\n"
"# investor.  ... Focus on financial\n"
"# implications, not general emotional tone.\n"
"#\n"
"#   <- character-for-character the ORIGINAL\n"
"#      docstring from cell 29\n"
"\n"
"print(len(compiled_mipro.predict.demos))\n"
"# -> 0"))
railcard(ax, 61.5, 39, 35.0, 19, "amber", "?", "So what did the search do?", [
    "10 trials, 6 demo candidates, 3 instruction candidates.",
    "None beat the starting point on the 51-example valset.",
    "MIPROv2 correctly kept the incumbent."], fs_body=7.5)

# the identity, as one line rather than two competing cards
card(ax, 61.5, 27.5, 35.0, 9.5, z=3, tint="rose", accent="rose")
ax.text(79.0, 34.0, "ChainOfThought  =  MIPROv2", fontsize=9.6, color=ROSE, ha="center",
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(79.0, 30.2, "74 / 102   =   72.55%   for both", fontsize=9.0, color=ROSE,
        ha="center", va="center", fontweight="bold", fontfamily=MONO, zorder=5)
strip(ax, 3.5, 16.0, 93.0, 8.6, "green", "+",
      "This is the honest reading: a zero-demo, unmodified-instruction ChainOfThought IS what "
      "MIPROv2 returned, so scoring identically to the CoT baseline is arithmetic, not luck. The "
      "optimizer did not fail - it reported that it could not beat the starting point.")
takeaway(ax, "Identical scores are a diagnostic, not a curiosity. When two runs tie exactly, check "
             "whether they are literally the same program before you explain the tie.",
         y=2.5, h=11, right_lines=["0 demos kept", "instruction unchanged", "74/102 both"])
save(fig, "08_why_mipro_tied.png")


# ══════════════════════════════════════════════ 09
fig, ax = canvas(13, 7.0); frame(ax)
header(ax, 9, "The notebook celebrates its own regression",
       "cell 55 computes the number correctly, then narrates it backwards", glyph="bug")
codeblock(ax, 3.5, 40, 55.0, 30, fs=8.8, step=1, step_tint="rose",
          title_text="cell 55 - the code, and what it printed", code=(
"improvement = mipro_score.score - baseline_predict_score.score\n"
"print(f\"Total improvement (baseline -> MIPROv2):\"\n"
"      f\" +{improvement:.1f} percentage points\")\n"
"\n"
"# Total improvement (baseline -> MIPROv2):\n"
"#     +-5.9 percentage points\n"
"# That's a -7.5% relative improvement\n"
"# ...without hand-writing a single new\n"
"#    prompt string."))
railcard(ax, 61.5, 52, 35.0, 18, "amber", "i", "The f-string bug", [
    "A literal '+' sits in front of {improvement:.1f}.",
    "The value is negative, so it renders '+-5.9'.",
    "Use {improvement:+.1f} and let the format spec own the sign."], fs_body=7.5)
railcard(ax, 61.5, 38, 35.0, 12.5, "rose", "!", "The wording bug", [
    "'Total improvement' and 'relative improvement' both describe a decline."], fs_body=7.5)
ax.text(3.5, 35.5, "Both numbers are arithmetically correct:", fontsize=8.6, color=NAVY,
        fontweight="bold", fontfamily=SANS, zorder=5)
for yy, lab, expr, val, tint in [(26.5, "absolute", "72.55 - 78.43", "-5.88 pts", "rose"),
                                 (17.5, "relative", "72.55 / 78.43 - 1", "-7.50 %", "rose")]:
    card(ax, 3.5, yy, 55.0, 8.0, z=3, accent=tint)
    ax.text(6.5, yy + 4.0, lab, fontsize=8.4, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(22.0, yy + 4.0, expr, fontsize=8.4, color=BODY, va="center", fontfamily=MONO, zorder=5)
    ax.text(55.0, yy + 4.0, val, fontsize=11.5, color=STRONG[tint], va="center", ha="right",
            fontweight="bold", fontfamily=SANS, zorder=5)
strip(ax, 61.5, 17.5, 35.0, 17.0, "green", "+",
      "The maths never lied. Only the sentence around it did - which is exactly how a regression "
      "ships to production unnoticed.")
takeaway(ax, "Never let prose assert a direction the number has not been checked for. Assert it in "
             "code: if improvement < 0, say 'regression', not 'improvement'.",
         y=2.5, h=11, right_lines=["'+-5.9' is a tell", "format spec owns the sign", "read your own output"])
save(fig, "09_celebrated_regression.png")


# ══════════════════════════════════════════════ 10
fig, ax = canvas(13, 6.8); frame(ax)
header(ax, 10, "One rate limit, two different percentages",
       "the BootstrapFewShot run lost a single example - and the score moved 0.75 pts", glyph="chain")
codeblock(ax, 3.5, 34, 55.0, 30, fs=8.6, step=1, step_tint="rose",
          title_text="cell 46 - both lines come from the SAME run", code=(
"Average Metric: 73.00 / 96 (76.0%): 94%| 96/102\n"
"ERROR dspy.utils.parallelizer: litellm.RateLimitError\n"
"  Rate limit reached ... TPM: Limit 6000,\n"
"  Used 5225, Requested 874\n"
"\n"
"Average Metric: 77.00 / 101 (76.2%): 100%| 102/102\n"
"INFO  dspy.evaluate: Average Metric: 77.0 / 102 (75.5%)\n"
"\n"
"BootstrapFewShot accuracy: 75.49%"))
ax.text(61.5, 61.0, "Same 77 correct answers, two denominators",
        fontsize=8.8, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
scorerow(ax, 61.5, 49.0, 35.0, 9.5, "progress bar", "77 / 101 survivors", 76.24, "amber", "", lw=19, big=10.5)
scorerow(ax, 61.5, 38.0, 35.0, 9.5, "final report", "77 / 102 devset", 75.49, "rose", "", lw=19, big=10.5)
ax.text(61.5, 35.0, "the one lost example is worth 0.75 points",
        fontsize=7.6, color=ROSE, fontweight="bold", fontfamily=SANS, zorder=5)
strip(ax, 3.5, 16.0, 93.0, 14.0, "amber", "i",
      "Evaluate counts an errored example as WRONG, not as missing - it stays in the denominator "
      "and contributes 0.0. So an infrastructure failure is indistinguishable from a bad "
      "prediction in the headline number. Here it is only 0.75 pts and does not change the "
      "conclusion, but the same mechanism silently decides close comparisons.")
takeaway(ax, "Distinguish 'the program was wrong' from 'the call never happened'. Report "
             "correct / scored / attempted, never a bare percentage.",
         y=2.5, h=11, right_lines=["76.24% survivors", "75.49% reported", "1 lost call"])
save(fig, "10_silent_denominator.png")


# ══════════════════════════════════════════════ 11
fig, ax = canvas(13, 7.2); frame(ax)
header(ax, 11, "The error analysis contradicts itself",
       "cell 40 describes one failure mode; cell 39 printed the opposite one", glyph="audit")
card(ax, 3.5, 42, 45.0, 34, z=3, accent="rose")
chip(ax, 8.0, 72.0, "rose", "M", r=2.0, fs=9)
ax.text(11.8, 72.0, "What the markdown claims  (cell 40)", fontsize=9.0, color=ROSE,
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
body(ax, 6.5, 66.0,
     "\"Neutral overclaim: model predicts neutral for anything without explicit up/down words.\"\n\n"
     "\"Domain-blindness ... often mispredicted as neutral.\"\n\n"
     "Two of the three named failure modes say the model over-predicts NEUTRAL.",
     width=56, fs=8.0, lh=2.9)
card(ax, 51.5, 42, 45.0, 34, z=3, accent="green")
chip(ax, 56.0, 72.0, "green", "D", r=2.0, fs=9)
ax.text(59.8, 72.0, "What the output shows  (cell 39)", fontsize=9.0, color=GREEN,
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(54.5, 66.2, "7 errors in the first 30 test examples; 5 printed:",
        fontsize=7.8, color=BODY, fontfamily=SANS, zorder=5)
for i, (g, pr, n, tint) in enumerate([("neutral", "positive", 4, "rose"),
                                      ("negative", "neutral", 1, "grey")]):
    yy = 60.0 - i * 7.6
    card(ax, 54.5, yy - 2.6, 39.0, 5.6, tint=tint, z=4)
    ax.text(57.0, yy + 0.2, f"gold {g}  ->  predicted {pr}", fontsize=8.0,
            color=STRONG[tint], va="center", fontweight="bold", fontfamily=MONO, zorder=6)
    ax.text(91.5, yy + 0.2, f"x{n}", fontsize=9.5, color=STRONG[tint], va="center",
            ha="right", fontweight="bold", fontfamily=SANS, zorder=6)
ax.text(54.5, 49.5, "4 of 5 are neutral gold read as POSITIVE - over-optimism,\n"
        "not the neutral-overclaim the markdown predicted.",
        fontsize=7.7, color=GREEN, fontfamily=SANS, zorder=5, va="top", linespacing=1.7)
strip(ax, 3.5, 30.0, 93.0, 9.0, "amber", "i",
      "It matters because cell 40 uses its diagnosis to argue what 'good examples in the prompt "
      "would fix' - the justification for reaching for a teleprompter in the next section. The "
      "demos that fix over-optimism are not the demos that fix over-neutrality.")
card(ax, 3.5, 21.5, 93.0, 6.2, z=3)
ax.text(6.0, 24.6, "Rule:  read the printed errors before you believe the narrative about them. "
        "Cell 40 says \"what we typically see\" - written from expectation, before the run.",
        fontsize=8.0, color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
takeaway(ax, "Error analysis written in advance is a hypothesis. Only the confusion counts printed "
             "by the run are evidence.",
         y=3.5, h=13, right_lines=["claimed: over-neutral", "observed: over-positive", "4 of 5 errors"])
save(fig, "11_error_analysis_contradiction.png")


# ══════════════════════════════════════════════ 12
fig, ax = canvas(13, 8.0); frame(ax)
header(ax, 12, "Which optimizer, and when",
       "the notebook runs rows 2 and 4 - the rest is the map around them", glyph="choose")
for x, t in [(3.5,"Optimizer"), (27.0,"What it searches"), (58.0,"Cost"), (70.0,"Reach for it when")]:
    ax.text(x, 80.5, t, fontsize=8.0, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)
opts = [
    ("LabeledFewShot", "grey", "Picks k demos straight from your labels. No LM calls.",
     "free", "You want a floor to beat before spending anything.", ""),
    ("BootstrapFewShot", "blue", "Self-generated traces filtered by the metric. Stops at the cap.",
     "~n calls", "First move on any task. Best optimizer here at 75.49%.", "<- run"),
    ("+ WithRandomSearch", "purple", "Runs Bootstrap many times over random subsets, keeps the best on a valset.",
     "10-50x", "Bootstrap helped and you have a valset plus budget.", ""),
    ("MIPROv2", "teal", "Proposes INSTRUCTIONS as well as demos; TPE surrogate over both.",
     "100s", "You suspect the wording itself is the bottleneck.", "<- run"),
    ("COPRO", "amber", "Coordinate ascent on instruction text only.", "100s",
     "You cannot use demos - long inputs or a tight token budget.", ""),
    ("BootstrapFinetune", "green", "Distils the compiled prompt into model weights.", "GPU",
     "Latency or per-call cost beats iteration speed.", ""),
]
y = 70
for nm, tint, what, cost, when, tag in opts:
    card(ax, 3.5, y, 93.0, 8.8, z=3, accent=tint)
    ax.text(6.5, y + 4.4, nm, fontsize=8.6, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    body(ax, 27.0, y + 5.5, what, width=42, fs=7.3, lh=2.3)
    ax.text(58.0, y + 4.4, cost, fontsize=7.8, color=NAVY, va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    body(ax, 70.0, y + 5.5, when, width=34, fs=7.3, lh=2.3)
    if tag:
        ax.text(94.5, y + 4.4, tag, fontsize=7.2, color=BLUE, va="center", ha="right",
                fontweight="bold", fontfamily=SANS, zorder=6)
    y -= 9.8
strip(ax, 3.5, 13.0, 93.0, 6.4, "amber", "+",
      "Engineering context: only BootstrapFewShot and MIPROv2 appear in this notebook. The other "
      "four rows are the surrounding family - do not claim notebook evidence for them.")
takeaway(ax, "Climb only when the rung below stops paying - and here, no rung paid. With 21 "
             "training examples and a 78.43% baseline, there was very little headroom to search.",
         y=2.0, h=9.5, right_lines=["start cheap", "measure", "be willing to stop"])
save(fig, "12_optimizer_landscape.png")


# ══════════════════════════════════════════════ 13
fig, ax = canvas(13, 6.8); frame(ax)
header(ax, 13, "The one mental model to keep",
       "DSPy is a compiler; the prompt is the assembly it emits", glyph="recap")
stages = [
    (3.5,  "blue",   "S", "Signature", "typed contract",
     "what goes in, what comes out - the docstring is the instruction"),
    (27.0, "purple", "M", "Module", "execution strategy",
     "Predict / ChainOfThought / ReAct, swapped without touching the signature"),
    (50.5, "amber",  "O", "Optimizer", "the compiler pass",
     "searches instructions and demos - and may return the program unchanged"),
    (74.0, "teal",   "f", "Metric", "definition of better",
     "the only thing search can see - every blind spot here is inherited"),
]
for x, tint, g, nm, sub, desc in stages:
    card(ax, x, 50, 22.5, 24, z=3, accent=tint)
    chip(ax, x + 4.2, 70.0, tint, g, r=2.1, fs=9.5)
    ax.text(x + 8.2, 70.0, nm, fontsize=9.6, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + 3.2, 64.8, sub, fontsize=7.8, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    body(ax, x + 3.2, 60.5, desc, width=30, fs=7.5, lh=2.8)
    if x < 74:
        arrow(ax, x + 22.8, 62.0, x + 26.7, 62.0, color=STRONG[tint], lw=2.0, ms=12)
card(ax, 3.5, 34, 93.0, 12.5, z=3, tint="indigo")
ax.text(6.5, 42.0, "and the rule this lecture actually taught:", fontsize=8.6, color=INDIGO,
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(6.5, 37.6, "Optimization is a hypothesis you test, not a step that always pays. "
        "Here the unoptimized baseline won by 2.94 points.",
        fontsize=9.2, color="#3B3F73", va="center", fontfamily=SANS, zorder=5)
for x, txt, tint in [(3.5, "Did the optimizer beat the baseline?", "teal"),
                     (35.7, "Is the metric measuring what I mean?", "rose"),
                     (67.9, "Did every example actually run?", "amber")]:
    card(ax, x, 22.0, 28.6, 8.4, z=3, accent=tint)
    chip(ax, x + 4.0, 26.2, tint, "?", r=1.8, fs=9)
    body(ax, x + 7.5, 27.5, txt, width=27, fs=7.6, lh=2.6)
takeaway(ax, "Program, do not prompt - but always keep the unoptimized baseline in the results "
             "table. It is the only thing that tells you whether the compiler earned its keep.",
         y=2.5, h=14.0, right_lines=["declare", "measure", "compile", "compare"])
save(fig, "13_mental_model.png")

print("all 13 written")
