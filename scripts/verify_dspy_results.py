"""Verify every number quoted in the L14 (DSPy) notes against the notebook's output.

Cells are located by CONTENT, never by index, so inserting a cell (e.g. the
Open-in-Colab badge) cannot silently invalidate this script.

Run:  python3 scripts/verify_dspy_results.py
"""
import json, re, pathlib, sys
from collections import Counter

NB = pathlib.Path(__file__).resolve().parents[1] / \
    "8. Agentic AI Systems/14. DSPy: Mathematical Prompt Optimization/L14_DSPy_Mathematical_Prompt_Optimization.ipynb"
cells = json.load(open(NB))["cells"]

def out(i):
    s = []
    for o in cells[i].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t: s.append("".join(t) if isinstance(t, list) else t)
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(s))

def find(*needles):
    hits = [i for i, c in enumerate(cells)
            if all(n in "".join(c.get("source", [])) for n in needles)]
    assert len(hits) == 1, f"expected 1 cell matching {needles}, found {hits}"
    return hits[0]

C = {
    "lm":       find("lm = dspy.LM("),
    "dist":     find("label_map = {0:", "Class distribution"),
    "splits":   find("def stratified_sample", "per_class=34"),
    "metric":   find("def sentiment_match"),
    "predict":  find("baseline_predict = dspy.Predict"),
    "cot":      find("baseline_cot = dspy.ChainOfThought"),
    "errors":   find("errors = []", "Errors in first 30"),
    "claims":   find("Neutral overclaim"),
    "boot":     find("bootstrap.compile(student=student"),
    "bootev":   find("bootstrap_score = evaluator(compiled_bootstrap)"),
    "mipro":    find("mipro = MIPROv2("),
    "miproins": find("compiled_mipro.predict.signature.instructions"),
    "miproev":  find("mipro_score = evaluator(compiled_mipro)"),
    "table":    find('results = {', "Predict (baseline)"),
    "chart":    find("Total improvement (baseline"),
    "wrapup":   find("The three things to remember"),
    "lcintro":  find("Section 8: LangChain Integration"),
}
print("resolved cells:", {k: v for k, v in sorted(C.items(), key=lambda kv: kv[1])})

ok = []
def chk(name, got, want, tol=0.01):
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"  {'PASS' if good else 'FAIL'}  {name:<48} computed {got:>8.2f}   notebook {want:>8.2f}")
def assert_(name, cond):
    ok.append(bool(cond)); print(f"  {'PASS' if cond else 'FAIL'}  {name}")

print(f"\n1. SETUP")
model = re.search(r'model="([^"]+)"', "".join(cells[C["lm"]]["source"])).group(1)
assert_(f"single LM throughout: {model}", model == "groq/llama-3.1-8b-instant")

print(f"\n2. DATASET & SPLITS")
neu, pos, neg, tot = 2298, 1091, 483, 3872
assert f"Counter({{'neutral': {neu}, 'positive': {pos}, 'negative': {neg}}})" in out(C["dist"])
chk("class counts sum to total", neu + pos + neg, tot)
chk("always-neutral on raw pool %", 100 * neu / tot, 59.35)
assert "Train: 21 | Val: 51 | Test: 102" in out(C["splits"])
chk("balanced-random floor on test set %", 100 / 3, 33.33)

print(f"\n3. THE FOUR MEASURED RESULTS  (all on the same 102-example test set)")
res = {}
for key, label, corr, pct in [("predict","Predict (baseline)",80,78.43),
                              ("cot","ChainOfThought",74,72.55),
                              ("bootev","BootstrapFewShot",77,75.49),
                              ("miproev","MIPROv2 (light)",74,72.55)]:
    t = out(C[key])
    assert f"{corr}.0 / 102" in t, f"{label}: '{corr}.0 / 102' not in output"
    chk(f"{label:24} {corr}/102", 100 * corr / 102, pct)
    res[label] = 100 * corr / 102

print(f"\n4. THE HEADLINE FINDING — every optimizer lost to the plain baseline")
base = res["Predict (baseline)"]
for label in ["ChainOfThought", "BootstrapFewShot", "MIPROv2 (light)"]:
    d = res[label] - base
    ok.append(d < 0)
    print(f"  {'PASS' if d < 0 else 'FAIL'}  {label:24} vs baseline: {d:+.2f} pts")
chk("best optimizer (BootstrapFewShot) still below baseline", base - res["BootstrapFewShot"], 2.94)
assert_("nothing reached the notebook's own 80% target line", max(res.values()) < 80)

print(f"\n5. WHY MIPROv2 == ChainOfThought EXACTLY")
t = out(C["miproins"])
orig_doc = "Classify the sentiment of a financial news headline from the perspective of an equity investor."
assert_("MIPROv2's chosen instruction == the original docstring", orig_doc in t)
assert_("MIPROv2 kept ZERO demos", "Number of demos: 0" in t)
assert_("so its score is identical to the CoT baseline",
        res["MIPROv2 (light)"] == res["ChainOfThought"])
print("      -> MIPROv2 searched and returned the unmodified starting program.")

print(f"\n6. THE NEGATIVE 'IMPROVEMENT' THE NOTEBOOK PRINTS")
t = out(C["chart"])
assert_("cell prints '+-5.9 percentage points' (f-string sign collision)",
        "+-5.9 percentage points" in t)
chk("that value = MIPROv2 - Predict", res["MIPROv2 (light)"] - base, -5.88, tol=0.02)
assert_("and prints a -7.5% relative 'improvement'", "-7.5% relative improvement" in t)
chk("relative change", (res["MIPROv2 (light)"] / base - 1) * 100, -7.50, tol=0.02)

print(f"\n7. TWO CLAIMS THE RESULTS CONTRADICT")
lc = "".join(cells[C["lcintro"]]["source"])
assert_("Section 8 claims 'we have an 80%+ accurate classifier'", "80%+ accurate" in lc)
wrappers = [i for i, c in enumerate(cells)
            if "def dspy_classify" in "".join(c.get("source", []))]
assert_("  ...but it wires compiled_mipro (72.55%, joint-worst) into LangChain",
        all("compiled_mipro" in "".join(cells[i]["source"]) for i in wrappers))
assert_(f"  (aside) dspy_classify is defined twice, in cells {wrappers} - cell "
        f"{wrappers[1]} repeats {wrappers[0]} verbatim then extends it", len(wrappers) == 2)
wu = "".join(cells[C["wrapup"]]["source"])
assert_("wrap-up claims '~60% to 80%+ accuracy'", "60% to 80%+" in wu)

print(f"\n8. BootstrapFewShot LOST ONE EXAMPLE TO A RATE LIMIT")
t = out(C["bootev"])
assert_("a RateLimitError was logged", "RateLimitError" in t)
assert_("progress bar shows 77.00 / 101 (survivors)", "77.00 / 101" in t)
assert_("final report divides by 102", "77.0 / 102 (75.5%)" in t)
chk("survivors-only score", 100 * 77 / 101, 76.24)
chk("gap created by the one lost example", 100*77/101 - 100*77/102, 0.75)

print(f"\n9. ERROR ANALYSIS vs THE PROSE ABOUT IT")
t = out(C["errors"])
c = Counter(re.findall(r"Gold: (\w+)\s+\|\s+Predicted: (\w+)", t))
chk("errors in first 30 test examples", int(re.findall(r"Errors in first 30: (\d+)", t)[0]), 7)
print("      printed error directions:", dict(c))
chk("gold=neutral -> predicted=positive", c.get(("neutral", "positive"), 0), 4)
assert_("prose claims the model over-predicts 'neutral'",
        "Neutral overclaim" in "".join(cells[C["claims"]]["source"]))
print("      -> 4 of the 5 printed errors are the exact opposite: over-optimism.")

print("\n" + ("ALL CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED"))
sys.exit(0 if all(ok) else 1)
