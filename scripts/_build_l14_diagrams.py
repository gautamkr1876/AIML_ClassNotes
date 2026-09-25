"""Diagrams for L14 - DSPy: Mathematical Prompt Optimization.

Every figure is built from `scripts/diagram_kit.py`. All code shown is real
notebook code; all numbers are the ones verified by
`scripts/verify_dspy_eval_arithmetic.py`.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagram_kit import *   # noqa
import diagram_kit as K

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "8. Agentic AI Systems", "14. DSPy: Mathematical Prompt Optimization",
                   "images")
set_output(os.path.abspath(OUT))


def strip(ax, x, y, w, h, tint, glyph, text, fs=7.9):
    """Coloured failure / note strip."""
    card(ax, x, y, w, h, tint=tint, accent=tint, z=3)
    chip(ax, x + 4.2, y + h / 2, tint, glyph, r=1.8, fs=8.5)
    n = wrap_chars(ax, w - 12.0, fs)
    import textwrap as tw
    lines = tw.wrap(text, n)
    lh = K._lineh(ax, fs, 1.7)
    yy = y + h / 2 + lh * (len(lines) - 1) / 2
    for ln in lines:
        ax.text(x + 7.6, yy, ln, fontsize=fs, color=STRONG[tint], va="center",
                fontfamily=SANS, zorder=5, fontweight="bold")
        yy -= lh


def evrow(ax, x, y, w, label_txt, val, frac, tint, note="", h=5.4, lw=27):
    """Evidence row: label | proportional bar | value | note."""
    card(ax, x, y, w, h, z=3)
    ax.text(x + 2.6, y + h / 2, label_txt, fontsize=8.0, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    bx, bw = x + lw, w - lw - 24.0
    ax.add_patch(K.FancyBboxPatch((bx, y + h / 2 - 1.15), bw, 2.3,
                 boxstyle="round,pad=0,rounding_size=1.1", fc="#EDF1F7",
                 ec="none", zorder=4))
    if frac > 0:
        ax.add_patch(K.FancyBboxPatch((bx, y + h / 2 - 1.15), max(bw * frac, 1.2), 2.3,
                     boxstyle="round,pad=0,rounding_size=1.1", fc=STRONG[tint],
                     ec="none", zorder=5))
    ax.text(bx + bw + 3.0, y + h / 2, val, fontsize=8.6, color=STRONG[tint],
            va="center", fontweight="bold", fontfamily=MONO, zorder=6)
    if note:
        ax.text(x + w - 2.6, y + h / 2, note, fontsize=7.2, color=FAINT,
                va="center", ha="right", fontfamily=SANS, zorder=6)


# ══════════════════════════════════════════════════ 01
fig, ax = canvas(13, 7.0)
frame(ax)
header(ax, 1, "You write the contract. DSPy writes the prompt.",
       "four lines of Python on the right produced every line on the left", glyph="dspy")

codeblock(ax, 3.5, 24, 45.5, 52, fs=9.2, step=1, step_tint="green",
          title_text="What you write  (cell 14)", code=(
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
          title_text="What DSPy sends  (cell 16, inspect_history)", code=(
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
      "The docstring became the objective. The field names became the [[ ## ## ]] slots. "
      "The desc= became the constraint. Nothing here was hand-tuned - it is compiled output.")

takeaway(ax, "A prompt is build output, not source code. The source is a typed signature; "
             "the prompt is what the adapter generates from it.",
         right_lines=["4 lines in", "19 lines out", "zero prompt edits"])
save(fig, "01_contract_not_prompt.png")


# ══════════════════════════════════════════════════ 02
fig, ax = canvas(13, 7.2)
frame(ax)
header(ax, 2, "Why hand-tuned prompts rot",
       "the problem DSPy exists to solve - read left column top to bottom", glyph="why")

rows = [
    ("1", "grey",  "Problem",
     "You need a financial-sentiment classifier. 'Fed signals patience on rate cuts' is "
     "neutral in English and negative for growth stocks."),
    ("2", "blue",  "Naive approach",
     "Write a prompt. Add rules. Paste in three examples that fixed the last bug. Ship the string."),
    ("3", "rose",  "Limitation",
     "The string is untyped, untested and unversioned. Swap the model and it silently degrades. "
     "Nobody can tell which of the 14 clauses is load-bearing."),
    ("4", "green", "Solution",
     "Declare inputs, outputs and a metric. Let a search procedure pick the demonstrations and "
     "wording against measured score. That search is the optimizer."),
    ("5", "amber", "Trade-off",
     "You now need a labelled dev set, a metric you trust, and an API budget - and the "
     "optimizer is only ever as honest as that metric."),
]
y = 68
for n, tint, head, txt in rows:
    card(ax, 4, y, 93, 10.4, z=3, accent=tint)
    stepbadge(ax, 9.0, y + 5.2, n, tint, r=2.0, fs=9.5)
    ax.text(13.5, y + 5.2, head, fontsize=10.0, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    body(ax, 31.0, y + 6.7, txt, width=84, fs=8.5, lh=2.7)
    y -= 11.6

takeaway(ax, "DSPy does not make prompts better by writing nicer English. It makes them "
             "measurable, so an algorithm can search them.",
         y=3.0, h=11.5, right_lines=["prompt -> parameter", "eyeball -> metric", "edit -> compile"])
save(fig, "02_why_prompts_rot.png")


# ══════════════════════════════════════════════════ 03
fig, ax = canvas(13, 8.0)
frame(ax)
header(ax, 3, "The four primitives",
       "every DSPy program is these four objects and nothing else", glyph="core")

cards = [
    (3.0,  "blue",   "S", "Signature", [
        "Typed I/O contract.", "Docstring = the instruction.",
        "desc= = per-field constraint.", "Says WHAT, never HOW."]),
    (27.2, "purple", "M", "Module", [
        "The strategy that runs it.",
        "Predict = one shot.",
        "ChainOfThought = + reasoning field.",
        "Swappable without touching the signature."]),
    (51.4, "teal",   "f", "Metric", [
        "(example, pred, trace) -> float.",
        "The only definition of 'better'.",
        "Exact match here; could be an LLM judge.",
        "Optimizer inherits all its blind spots."]),
    (75.6, "amber",  "O", "Optimizer", [
        "Search over demos and instructions.",
        "Reads the metric, not your opinion.",
        "compile() returns a NEW module.",
        "a.k.a. teleprompter."]),
]
for x, tint, g, head, lines in cards:
    railcard(ax, x, 45, 21.8, 30, tint, g, head, lines, fs_head=9.8, fs_body=7.5)

codeblock(ax, 3.0, 17, 94.4, 23, fs=9.6, step=5, step_tint="indigo",
          title_text="How they compose  (cells 23, 26, 32)", code=(
"def sentiment_match(example, pred, trace=None):      # METRIC\n"
"    return float(example.sentiment.strip().lower() == predicted)\n"
"\n"
"student   = dspy.ChainOfThought(FinancialSentiment)  # MODULE wrapping a SIGNATURE\n"
"bootstrap = BootstrapFewShot(metric=sentiment_match, max_bootstrapped_demos=4)\n"
"compiled  = bootstrap.compile(student=student, trainset=trainset)   # OPTIMIZER"))

takeaway(ax, "Signature = what. Module = how. Metric = better. Optimizer = search. "
             "Change any one without rewriting the other three.",
         y=2.0, h=13.0, right_lines=["declare", "strategise", "measure", "search"])
save(fig, "03_four_primitives.png")


# ══════════════════════════════════════════════════ 04
fig, ax = canvas(13, 8.2)
frame(ax)
header(ax, 4, "Predict vs ChainOfThought",
       "same signature, one extra output field - and it changed everything", glyph="module")

codeblock(ax, 3.5, 46, 44.0, 29, fs=9.4, step=1, step_tint="blue",
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

codeblock(ax, 53.0, 46, 43.5, 29, fs=9.4, step=2, step_tint="purple",
          title_text="dspy.ChainOfThought -> the adapter inserts `reasoning` first", code=(
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

ax.text(3.5, 40.3, "Measured on the same 102-example test set  (cells 27, 28)",
        fontsize=8.8, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
evrow(ax, 3.5, 33.2, 93.0, "Predict    accuracy", "79.41%", 0.7941, "green",
      "81 / 102   all scored", lw=30)
evrow(ax, 3.5, 26.8, 93.0, "ChainOfThought  accuracy", "71.57%", 0.7157, "rose",
      "73 / 102   4 never scored", lw=30)
evrow(ax, 3.5, 20.4, 93.0, "Predict    wall clock", "6.3 s", 0.091, "green",
      "16.20 it/s", lw=30)
evrow(ax, 3.5, 14.0, 93.0, "ChainOfThought  wall clock", "69.0 s", 1.0, "rose",
      "1.48 it/s  = 11x slower", lw=30)

takeaway(ax, "Reasoning is not free: it multiplies output tokens, and on a 6000 TPM free tier "
             "those tokens are what throttled the run. See diagram 10.",
         right_lines=["+1 field", "11x the time", "-7.84 points"])
save(fig, "04_predict_vs_cot.png")


# ══════════════════════════════════════════════════ 05
fig, ax = canvas(13, 7.8)
frame(ax)
header(ax, 5, "The metric is the whole specification",
       "an optimizer cannot want anything your metric does not measure", glyph="f(x)")

codeblock(ax, 3.5, 30, 52.0, 44, fs=8.6, step=1, step_tint="teal",
          title_text="cell 23 - the real metric", code=(
"def sentiment_match(example, pred, trace=None):\n"
'    """Return 1.0 if predicted sentiment matches\n'
'    gold, else 0.0. Robust to case/whitespace."""\n'
"    gold = example.sentiment.strip().lower()\n"
"    predicted = (pred.sentiment or \"\").strip().lower()\n"
"\n"
"    # Handle verbosity like 'Positive.' or\n"
"    # 'the sentiment is negative'\n"
"    for label in [\"positive\", \"negative\", \"neutral\"]:\n"
"        if label in predicted:\n"
"            predicted = label\n"
"            break\n"
"\n"
"    return float(gold == predicted)"))

railcard(ax, 59.0, 52, 37.5, 22, "teal", "f", "The contract", [
    "Takes (example, pred, trace).",
    "Returns a float - 1.0 / 0.0 here.",
    "trace is non-None during compilation, so you can score harder at compile time.",
    "Evaluate averages it; the optimizer maximises it."], fs_body=7.4)

railcard(ax, 59.0, 30, 37.5, 20, "rose", "!", "The trap in this one", [
    "`if label in predicted` is a SUBSTRING test.",
    "\"not positive\" contains \"positive\" -> scored 1.0.",
    "First match wins, so \"negative, not positive\" -> negative by list order.",
    "Verified: cells 24-25 only test the easy cases."], fs_body=7.4)

strip(ax, 3.5, 15.5, 93.0, 10.0, "amber", "i",
      "Cell 24 asserts Prediction('Positive.') scores 1.0 and cell 25 asserts 'negative' vs gold "
      "'positive' scores 0.0. Both pass. Neither probes the substring hole - and a lenient metric "
      "does not just mis-report, it teaches the optimizer to prefer hedged, verbose answers.")

takeaway(ax, "Write the adversarial metric test before you write the optimizer call. "
             "A metric bug is silently amplified by every compile step that follows.",
         right_lines=["metric = spec", "lenient = wrong", "test it first"])
save(fig, "05_metric_contract.png")


# ══════════════════════════════════════════════════ 06
fig, ax = canvas(13, 7.0)
frame(ax)
header(ax, 6, "Splitting data for an optimizer, not a model",
       "3,872 headlines -> 21 / 51 / 102, and each split has a different job", glyph="data")

ax.text(3.5, 76.5, "Raw Financial PhraseBank  (cell 9)  -  heavily imbalanced",
        fontsize=8.8, color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
bx, bw = 3.5, 93.0
segs = [("neutral", 2298, "grey"), ("positive", 1091, "blue"), ("negative", 483, "rose")]
x = bx
for name, n, tint in segs:
    w = bw * n / 3872
    card(ax, x, 66.5, w - 0.5, 7.2, tint=tint, accent=tint, z=3)
    ax.text(x + (w - 0.5) / 2, 71.4, name, fontsize=8.0, color=STRONG[tint],
            ha="center", va="center", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x + (w - 0.5) / 2, 68.4, f"{n}   {100*n/3872:.1f}%", fontsize=7.4,
            color=STRONG[tint], ha="center", va="center", fontfamily=MONO, zorder=5)
    x += w
strip(ax, 3.5, 57.5, 93.0, 6.6, "rose", "!",
      "Always answering 'neutral' scores 2298/3872 = 59.35% on this pool. Any accuracy you "
      "report must beat 59.35% before it means anything at all.")

sp = [(3.5,  "blue",  "trainset", "21", "7 per class", "7 pos / 7 neg / 7 neu",
       "Source of demonstrations. The optimizer runs the teacher over these and keeps the traces that score 1.0."),
      (35.7, "purple","valset", "51", "17 per class", "17 pos / 17 neg / 17 neu",
       "Scores candidate programs during search. Never used in this notebook - BootstrapFewShot does not take a valset."),
      (67.9, "teal",  "testset", "102", "34 per class", "34 pos / 34 neg / 34 neu",
       "Final, touched once. All three evaluations in cells 27/28/34 run against this.")]
for x0, tint, nm, n, per, cls, desc in sp:
    card(ax, x0, 28.0, 28.6, 25.5, z=3)
    chip(ax, x0 + 4.4, 49.2, tint, nm[0].upper(), r=2.1, fs=9)
    ax.text(x0 + 8.4, 49.8, nm, fontsize=9.6, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(x0 + 8.4, 46.6, per, fontsize=7.3, color=FAINT, va="center",
            fontfamily=SANS, zorder=5)
    ax.text(x0 + 25.0, 48.4, n, fontsize=17, color=STRONG[tint], va="center",
            ha="right", fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(x0 + 3.2, 42.0, cls, fontsize=7.6, color=STRONG[tint], va="center",
            fontfamily=MONO, zorder=5)
    body(ax, x0 + 3.2, 37.5, desc, width=40, fs=7.6, lh=2.7)

strip(ax, 3.5, 19.0, 93.0, 6.2, "amber", "i",
      "58 of the 483 negatives (12.0%) are consumed by the three splits - the scarce class is "
      "what caps how large a balanced split can get.")

takeaway(ax, "Stratify, or your metric measures the class prior instead of the program. "
             "Train is a demo pool here - not gradient data.",
         right_lines=["train = demos", "val = search", "test = once"])
save(fig, "06_data_splits.png")


# ══════════════════════════════════════════════════ 07
fig, ax = canvas(13, 7.2)
frame(ax)
header(ax, 7, "BootstrapFewShot, step by step",
       "the model writes its own few-shot examples - and the metric decides which survive",
       glyph="compile")

steps = [
    (3.0,  "blue",   "1", "Teacher runs",
     "An unoptimized copy of the module runs on a training example and emits a full trace: reasoning + answer."),
    (27.2, "teal",   "2", "Metric filters",
     "sentiment_match scores the trace against the gold label. Score < 1.0 and the trace is thrown away."),
    (51.4, "green",  "3", "Keep k traces",
     "Survivors become demonstrations. Capped by max_bootstrapped_demos=4."),
    (75.6, "purple", "4", "Student compiled",
     "The demos are baked into the student's prompt. compile() returns a new module; the original is untouched."),
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
          title_text="cell 32 - and what it actually reported", code=(
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

takeaway(ax, "BootstrapFewShot is not a search - it is a filter that stops at the cap. "
             "BootstrapFewShotWithRandomSearch is the version that actually searches.",
         right_lines=["4 of 21 used", "0.2s compile", "first-fit, not best-fit"])
save(fig, "07_bootstrap_algorithm.png")


# ══════════════════════════════════════════════════ 08
fig, ax = canvas(13, 7.0)
frame(ax)
header(ax, 8, "What compilation actually produced",
       "the artifact is a list of demonstrations - and they carry the teacher's reasoning", glyph="artifact")

codeblock(ax, 3.5, 20, 52.0, 54, fs=9.2, step=1, step_tint="purple",
          title_text="cell 33 - inspecting the compiled program", code=(
"print(len(compiled_bootstrap.predict.demos))\n"
"# -> 4\n"
"\n"
"demo = compiled_bootstrap.predict.demos[0]\n"
"print(demo.headline)\n"
"print(demo.reasoning)\n"
"print(demo.sentiment)\n"
"\n"
"# headline:  The fair value of the company's\n"
"#   investment properties went down to EUR\n"
"#   2.768 billion at the end of 2009 from\n"
"#   EUR 2.916 billion a year earlier.\n"
"#\n"
"# reasoning: The decrease in the fair value\n"
"#   ... suggests a potential decrease in the\n"
"#   company's assets and possibly its overall\n"
"#   value, which could negatively impact the\n"
"#   stock price.\n"
"#\n"
"# sentiment: negative"))

railcard(ax, 59.0, 49, 37.5, 25, "green", "+", "Why the reasoning matters", [
    "A plain few-shot demo is just (input, label).",
    "A bootstrapped demo is (input, REASONING, label).",
    "The student sees how a successful trace was argued, not only what it concluded.",
    "That is the whole trick - the model supervises itself."], fs_body=7.6)

railcard(ax, 59.0, 20, 37.5, 25, "amber", "i", "What a compiled program IS", [
    "Not new weights. The LM is untouched.",
    "A module + 4 demo objects serialised into the prompt.",
    "save() / load() persists it as JSON - version it like code.",
    "Swap the LM and the demos may no longer be optimal."], fs_body=7.6)

takeaway(ax, "Compilation in DSPy writes prompt content, never model weights. The deliverable "
             "is a JSON blob of demonstrations that you should check into version control.",
         y=3.0, h=12.5, right_lines=["4 demos", "reasoning included", "weights unchanged"])
save(fig, "08_compiled_artifact.png")


# ══════════════════════════════════════════════════ 09
fig, ax = canvas(13, 7.8)
frame(ax)
header(ax, 9, "The comparison this notebook cannot make",
       "three programs, three different denominators - only one number is trustworthy",
       glyph="!! ")

ax.text(3.5, 80.0, "Program", fontsize=8.2, color=FAINT, fontweight="bold",
        fontfamily=SANS, zorder=5)
ax.text(34.0, 80.0, "Examples actually scored (out of 102)", fontsize=8.2, color=FAINT,
        fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(94.0, 80.0, "Reported", fontsize=8.2, color=FAINT, fontweight="bold",
        ha="right", fontfamily=SANS, zorder=5)

rows = [
    ("dspy.Predict", "cell 27", 102, "79.41%", "green", "complete - 0 errors, 6.3s"),
    ("dspy.ChainOfThought", "cell 28", 98, "71.57%", "amber", "4 examples scored 0.0 by rate limit"),
    ("BootstrapFewShot", "cell 34", 49, "none", "rose", "raised at 59/102 - never printed a score"),
    ("  ^ survivors only", "38 / 49", 49, "77.55%", "grey", "what you would be tempted to quote - 53 examples missing"),
]
y = 66
for nm, cell, scored, rep, tint, note in rows:
    card(ax, 3.5, y, 93.0, 11.2, z=3, accent=tint)
    ax.text(7.0, y + 7.6, nm, fontsize=9.6, color=NAVY, va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(7.0, y + 3.4, cell, fontsize=7.4, color=FAINT, va="center",
            fontfamily=SANS, zorder=5)
    bx, bw = 34.0, 44.0
    ax.add_patch(K.FancyBboxPatch((bx, y + 4.6), bw, 3.0,
                 boxstyle="round,pad=0,rounding_size=1.4", fc="#EDF1F7", ec="none", zorder=4))
    ax.add_patch(K.FancyBboxPatch((bx, y + 4.6), bw * scored / 102.0, 3.0,
                 boxstyle="round,pad=0,rounding_size=1.4", fc=STRONG[tint], ec="none", zorder=5))
    ax.text(bx + bw + 2.5, y + 6.1, f"{scored}/102", fontsize=8.4, color=STRONG[tint],
            va="center", fontweight="bold", fontfamily=MONO, zorder=6)
    ax.text(bx, y + 1.9, note, fontsize=7.3, color=BODY, va="center",
            fontfamily=SANS, zorder=6)
    ax.text(94.0, y + 5.9, rep, fontsize=13.0, color=STRONG[tint], va="center",
            ha="right", fontweight="bold", fontfamily=SANS, zorder=6)
    y -= 12.4

strip(ax, 3.5, 17.5, 93.0, 8.5, "rose", "!",
      "The notebook's arc is 'optimization beats the baseline'. What it actually recorded is the "
      "opposite and worse: the only complete run is the UNOPTIMIZED one, and the optimized run "
      "never produced a number at all. The bottom row is the trap - 77.55% looks like a near-miss "
      "against 79.41%, but it is a different, easier, half-sized test set.")

takeaway(ax, "A percentage without its denominator is not a measurement. Three runs over three "
             "different subsets are three experiments, not one comparison.",
         right_lines=["102 vs 98 vs 49", "one clean run", "zero valid deltas"])
save(fig, "09_broken_benchmark.png")


# ══════════════════════════════════════════════════ 10
fig, ax = canvas(13, 7.4)
frame(ax)
header(ax, 10, "How a rate limit became an accuracy drop",
       "the failure chain behind ChainOfThought's 71.57% - no stage raised a visible error",
       glyph="chain")

chain = [
    (3.0,  "purple", "1", "CoT adds a field",
     "Every call now emits a reasoning paragraph before the label."),
    (22.3, "blue",   "2", "Tokens multiply",
     "Throughput falls from 16.20 to 1.48 it/s - the run takes 69.0s instead of 6.3s."),
    (41.6, "amber",  "3", "TPM ceiling hit",
     "Groq free tier: 6000 tokens/min. Logged 'Used 5571, Requested 805'."),
    (60.9, "rose",   "4", "4 examples error",
     "litellm.RateLimitError. The parallelizer logs it and moves on."),
    (80.2, "rose",   "5", "Scored as 0.0",
     "They stay in the denominator. 4 free wrong answers."),
]
for x, tint, n, head, txt in chain:
    card(ax, x, 56, 17.8, 20, z=3, accent=tint)
    stepbadge(ax, x + 3.6, 72.6, n, tint, r=1.9, fs=9.0)
    ax.text(x + 7.0, 72.6, head, fontsize=8.2, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    body(ax, x + 3.0, 67.0, txt, width=26, fs=7.5, lh=2.7)
    if x < 80:
        arrow(ax, x + 18.1, 65.5, x + 22.2, 65.5, color=STRONG[tint], lw=1.8, ms=10)

codeblock(ax, 3.0, 20, 55.0, 29, fs=9.0, step=6, step_tint="rose",
          title_text="Both of these printed from the SAME cell (28)", code=(
"Average Metric: 73.00 / 98 (74.5%): 100%|####| 102/102\n"
"INFO dspy.evaluate: Average Metric: 73.0 / 102 (71.6%)\n"
"Baseline ChainOfThought accuracy: 71.57%\n"
"\n"
"# 73 correct in both lines.\n"
"# The progress bar divides by survivors.\n"
"# The final report divides by the full devset."))

ax.text(61.0, 51.7, "Decomposing the 7.84-point gap vs Predict", fontsize=8.8,
        color=NAVY, fontweight="bold", fontfamily=SANS, zorder=5)
for yy, lab, sub, val, tint in [
        (40.0, "Predict", "81 / 102  -  every example ran", "79.41%", "green"),
        (30.0, "CoT, survivors only", "73 / 98  -  the progress-bar number", "74.49%", "amber"),
        (20.0, "CoT, as reported", "73 / 102  -  the 4 errors count as wrong", "71.57%", "rose")]:
    card(ax, 61.0, yy, 35.5, 9.0, z=3, accent=tint)
    ax.text(64.0, yy + 5.9, lab, fontsize=8.2, color=NAVY, va="center",
            fontweight="bold", fontfamily=SANS, zorder=5)
    ax.text(64.0, yy + 2.6, sub, fontsize=6.9, color=FAINT, va="center",
            fontfamily=SANS, zorder=5)
    ax.text(94.5, yy + 4.5, val, fontsize=12.5, color=STRONG[tint], va="center",
            ha="right", fontweight="bold", fontfamily=SANS, zorder=5)

takeaway(ax, "Distinguish 'the program was wrong' from 'the call never happened'. Evaluate "
             "conflates them: 2.92 of the 7.84 lost points are infrastructure, 4.92 are real.",
         y=3.0, h=12.5, right_lines=["74.49% survivors", "71.57% reported", "same 73 correct"])
save(fig, "10_silent_denominator.png")


# ══════════════════════════════════════════════════ 11
fig, ax = canvas(13, 7.2)
frame(ax)
header(ax, 11, "The error analysis contradicts itself",
       "cell 30 describes one failure mode; cell 29 printed the opposite one", glyph="audit")

card(ax, 3.5, 42, 45.0, 34, z=3, accent="rose")
chip(ax, 8.0, 72.0, "rose", "M", r=2.0, fs=9)
ax.text(11.8, 72.0, "What the markdown claims  (cell 30)", fontsize=9.0, color=ROSE,
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
body(ax, 6.5, 66.0,
     "\"Neutral overclaim: model predicts neutral for anything without explicit "
     "up/down words.\"\n\n"
     "\"Domain-blindness ... often mispredicted as neutral.\"\n\n"
     "Two of the three named failure modes are the model over-predicting NEUTRAL.",
     width=56, fs=8.0, lh=2.9)

card(ax, 51.5, 42, 45.0, 34, z=3, accent="green")
chip(ax, 56.0, 72.0, "green", "D", r=2.0, fs=9)
ax.text(59.8, 72.0, "What the output shows  (cell 29)", fontsize=9.0, color=GREEN,
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
      "This matters beyond tidiness: cell 30 uses those claimed patterns to argue what "
      "'good examples in the prompt would fix'. Bootstrapping then selects demos against the "
      "metric - so a wrong diagnosis here sends you tuning for a failure mode you do not have.")

card(ax, 3.5, 21.5, 93.0, 6.2, z=3)
ax.text(6.0, 24.6, "Rule:  read the printed errors before you believe the narrative about them. "
        "Cell 30 says \"what we typically see\" - it was written before the run, not after it.",
        fontsize=8.0, color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)

takeaway(ax, "Error analysis written in advance is a hypothesis. Only the confusion counts "
             "printed by the run are evidence.",
         y=3.5, h=13, right_lines=["claimed: over-neutral", "observed: over-positive", "4 of 5 errors"])
save(fig, "11_error_analysis_contradiction.png")


# ══════════════════════════════════════════════════ 12
fig, ax = canvas(13, 8.0)
frame(ax)
header(ax, 12, "Which optimizer, and when",
       "the notebook uses only the second row - the rest is the map around it", glyph="choose")

ax.text(3.5, 80.5, "Optimizer", fontsize=8.0, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(27.0, 80.5, "What it searches", fontsize=8.0, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(58.0, 80.5, "Cost", fontsize=8.0, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(70.0, 80.5, "Reach for it when", fontsize=8.0, color=FAINT, fontweight="bold", fontfamily=SANS, zorder=5)

opts = [
    ("LabeledFewShot", "grey", "Picks k demos straight from your labels. No LM calls.",
     "free", "You already have gold examples and want a floor to beat.", False),
    ("BootstrapFewShot", "blue", "Self-generated traces, filtered by the metric. Stops at the cap.",
     "~n calls", "First move on any task. This is the notebook's choice.", True),
    ("+ WithRandomSearch", "purple", "Runs Bootstrap many times over random subsets, keeps the best on a valset.",
     "10-50x", "Bootstrap helped and you have a valset plus budget.", False),
    ("MIPROv2", "teal", "Proposes INSTRUCTIONS as well as demos; Bayesian search over both.",
     "100s", "Demos plateaued and the wording itself is the bottleneck.", False),
    ("COPRO", "amber", "Coordinate ascent on instruction text only.", "100s",
     "You cannot use demos - long inputs or a strict token budget.", False),
    ("BootstrapFinetune", "green", "Distils the compiled prompt into model weights.", "GPU",
     "Latency or per-call cost matters more than iteration speed.", False),
]
y = 70
for nm, tint, what, cost, when, here in opts:
    card(ax, 3.5, y, 93.0, 8.8, z=3, accent=tint)
    ax.text(6.5, y + 4.5, nm, fontsize=8.6, color=STRONG[tint], va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    body(ax, 27.0, y + 5.6, what, width=42, fs=7.3, lh=2.3)
    ax.text(58.0, y + 4.5, cost, fontsize=7.8, color=NAVY, va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    body(ax, 70.0, y + 5.6, when, width=36, fs=7.3, lh=2.3)
    if here:
        ax.text(94.5, y + 4.4, "<- used here", fontsize=7.2, color=BLUE, va="center",
                ha="right", fontweight="bold", fontfamily=SANS, zorder=6)
    y -= 9.8

strip(ax, 3.5, 13.0, 93.0, 6.4, "amber", "+",
      "Engineering context: only BootstrapFewShot appears in this notebook. The other five rows "
      "are the DSPy optimizer family you would reach for next - do not claim notebook evidence for them.")

takeaway(ax, "Climb the ladder only when the rung below stops paying. Every step up multiplies "
             "LM calls, and none of them can outrun a bad metric.",
         y=2.0, h=9.5, right_lines=["start cheap", "measure", "then escalate"])
save(fig, "12_optimizer_landscape.png")


# ══════════════════════════════════════════════════ 13
fig, ax = canvas(13, 6.8)
frame(ax)
header(ax, 13, "The one mental model to keep",
       "DSPy is a compiler; the prompt is the assembly it emits", glyph="recap")

stages = [
    (3.5,  "blue",   "S", "Signature", "typed contract",
     "what goes in, what comes out - and the docstring is the instruction"),
    (27.0, "purple", "M", "Module", "execution strategy",
     "Predict / ChainOfThought / ReAct, swapped without touching the signature"),
    (50.5, "teal",   "f", "Metric", "definition of better",
     "the only thing search can see - every blind spot here is inherited"),
    (74.0, "amber",  "O", "Optimizer", "the compiler pass",
     "emits demos and instructions, and returns a brand new module"),
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
ax.text(6.5, 42.0, "and the rule that outranks all four:", fontsize=8.6, color=INDIGO,
        va="center", fontweight="bold", fontfamily=SANS, zorder=5)
ax.text(6.5, 37.6, "An optimizer maximises your metric on the examples it actually scored. "
        "Check both halves of that sentence before you believe a number.",
        fontsize=9.2, color="#3B3F73", va="center", fontfamily=SANS, zorder=5)

for x, txt, tint in [(3.5, "Is the metric measuring what I mean?", "teal"),
                     (35.7, "Did every example actually run?", "rose"),
                     (67.9, "Is the denominator the same across runs?", "amber")]:
    card(ax, x, 22.0, 28.6, 8.4, z=3, accent=tint)
    chip(ax, x + 4.0, 26.2, tint, "?", r=1.8, fs=9)
    body(ax, x + 7.5, 27.5, txt, width=27, fs=7.6, lh=2.6)

takeaway(ax, "Program, do not prompt - but only after you can measure. DSPy turns prompt "
             "engineering into a search problem, and search is only as good as its objective.",
         y=2.5, h=14.0, right_lines=["declare", "measure", "compile", "verify"])
save(fig, "13_mental_model.png")

print("all 13 written")
