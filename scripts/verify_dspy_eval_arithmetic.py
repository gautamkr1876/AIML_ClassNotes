"""Verify every number quoted in the L14 (DSPy) notes against the notebook's output.

Cells are located by CONTENT, not by index, so inserting a cell (e.g. the
Open-in-Colab badge) cannot silently invalidate this script.

Run:  python3 scripts/verify_dspy_eval_arithmetic.py
"""
import json, re, pathlib, sys
from collections import Counter

NB = pathlib.Path(__file__).resolve().parents[1] / \
    "8. Agentic AI Systems/14. DSPy: Mathematical Prompt Optimization/DSPY_liveclass.ipynb"
cells = json.load(open(NB))["cells"]

def out(i):
    s = []
    for o in cells[i].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t: s.append("".join(t) if isinstance(t, list) else t)
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(s))

def find(*needles):
    """Index of the one cell whose source contains every needle."""
    hits = [i for i, c in enumerate(cells)
            if all(n in "".join(c.get("source", [])) for n in needles)]
    assert len(hits) == 1, f"expected 1 cell matching {needles}, found {hits}"
    return hits[0]

C = {
    "dist":      find("label_map = {0:", "Class distribution"),
    "splits":    find("def stratified_sample", "per_class=34"),
    "metric":    find("def sentiment_match"),
    "predict":   find("baseline_predict = dspy.Predict"),
    "cot":       find("baseline_cot = dspy.ChainOfThought"),
    "errors":    find("errors = []", "Errors in first 30"),
    "claims":    find("Neutral overclaim"),
    "compile":   find("bootstrap.compile(student=student"),
    "evalcomp":  find("bootstrap_score = evaluator(compiled_bootstrap)"),
}
print("resolved cell indices:", {k: v for k, v in sorted(C.items(), key=lambda kv: kv[1])})

ok = []
def chk(name, got, want, tol=0.01):
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"  {'PASS' if good else 'FAIL'}  {name:<46} computed {got:>8.2f}   notebook {want:>8.2f}")

print(f"\n1. DATASET  (cell {C['dist']})")
neu, pos, neg, tot = 2298, 1091, 483, 3872
assert f"Counter({{'neutral': {neu}, 'positive': {pos}, 'negative': {neg}}})" in out(C["dist"])
chk("neutral+positive+negative == total", neu + pos + neg, tot)
chk("majority-class baseline %  (2298/3872)", 100 * neu / tot, 59.35)
print("      -> 'always predict neutral' already scores 59.35% on the raw pool.")

print(f"\n2. SPLITS  (cell {C['splits']})  stratified, 3 classes")
assert "Train: 21 | Val: 51 | Test: 102" in out(C["splits"])
for nm, per, n in [("train", 7, 21), ("val", 17, 51), ("test", 34, 102)]:
    chk(f"{nm}: 3 x {per}", 3 * per, n)
chk("balanced-random floor on the test set %", 100 / 3, 33.33)
chk("negatives consumed (7+17+34)", 7 + 17 + 34, 58)
print(f"      -> 58 of 483 negatives = {100*58/483:.1f}% of the scarcest class.")

print(f"\n3. BASELINE  dspy.Predict  (cell {C['predict']})")
t = out(C["predict"])
assert "81.00 / 102" in t and "79.41" in t
chk("81/102", 100 * 81 / 102, 79.41)
chk("rate-limit errors in this cell", t.count("RateLimitError"), 0)
print("      -> 102/102 scored, zero errors, 6.3s at 16.20 it/s. The only clean run.")

print(f"\n4. BASELINE  dspy.ChainOfThought  (cell {C['cot']})")
t = out(C["cot"])
errs = len(set(re.findall(r"Error for Example\(\{'headline': (?:'|\").{0,60}", t)))
assert "73.00 / 98 (74.5%)" in t, "progress-bar figure moved"
assert "Average Metric: 73.0 / 102 (71.6%)" in t, "final INFO figure moved"
chk("progress bar  73/98  (survivors only)", 100 * 73 / 98, 74.49)
chk("final report   73/102 (errors == 0.0)", 100 * 73 / 102, 71.57)
chk("distinct examples lost to rate limits", errs, 4)
chk("102 - 98 == examples never scored", 102 - 98, errs)
chk("points of the drop that are pure infra", 100*73/98 - 100*73/102, 2.92)
chk("real deficit vs Predict (79.41 - 74.49)", 79.41 - 100 * 73 / 98, 4.92)
chk("headline gap Predict - CoT", 79.41 - 71.57, 7.84)
print("      -> the SAME cell prints 74.5% and 71.6%. Same 73 correct; different denominator.")
print(f"      -> CoT 69.0s vs Predict 6.3s = {69.0/6.3:.1f}x slower -> it caused its own throttling.")

print(f"\n5. BootstrapFewShot  (cells {C['compile']}, {C['evalcomp']})")
assert "Bootstrapped 4 full traces after 4 examples" in out(C["compile"])
print("      compile: 4 demos accepted from the first 4 of 21 train examples (19%), 0.2s.")
chk("train examples never attempted", 21 - 4, 17)
t = out(C["evalcomp"])
last = re.findall(r"Average Metric: ([\d.]+) / (\d+) \(([\d.]+)%\)", t)[-1]
chk("last survivor score 38/49", 100 * 38 / 49, float(last[2]), tol=0.06)
chk("progress reached /102", int(re.findall(r"\| (\d+)/102", t)[-1]), 59)
chk("distinct rate-limit failures", len(set(re.findall(r"Error for Example\(\{'headline': (?:'|\").{0,60}", t))), 10)
raised = cells[C["evalcomp"]]["outputs"][-1].get("output_type") == "error"
ok.append(raised)
print(f"  {'PASS' if raised else 'FAIL'}  cell terminates in an exception                "
      f"{cells[C['evalcomp']]['outputs'][-1].get('evalue','')}")
print("      -> no BootstrapFewShot accuracy was ever printed.")

print("\n6. THE COMPARISON THE NOTEBOOK IMPLIES")
print("      Predict            79.41%   81/102   complete")
print("      ChainOfThought     71.57%   73/102   4 examples scored 0.0 by rate limit")
print("      BootstrapFewShot      --    died at 59/102, 38/49 survivors (77.55%)")
print("      Only row 1 is trustworthy. 77.55% vs 79.41% is not a comparison.")

print(f"\n7. ERROR ANALYSIS  (cell {C['errors']} vs the cell-{C['claims']} markdown)")
t = out(C["errors"])
c = Counter(re.findall(r"Gold: (\w+)\s+\|\s+Predicted: (\w+)", t))
chk("errors found in first 30 test examples", int(re.findall(r"Errors in first 30: (\d+)", t)[0]), 7)
print("      printed error directions:", dict(c))
chk("gold=neutral -> predicted=positive", c.get(("neutral", "positive"), 0), 4)
print(f"      -> cell {C['claims']} claims the model over-predicts 'neutral'. 4 of the 5 printed")
print("         errors are the exact opposite.")

print("\n8. PROVENANCE")
decl = find("lm = dspy.LM(")
model = re.search(r'model="([^"]+)"', "".join(cells[decl]["source"])).group(1)
errmodels = set(re.findall(r"\[([a-z0-9.\-]+)\] litellm", out(C["cot"]) + out(C["evalcomp"])))
ok.append(model not in errmodels)
print(f"  {'PASS' if model not in errmodels else 'FAIL'}  declared model differs from the one in the errors")
print(f"      cell {decl} declares : {model}")
print(f"      errors name          : {', '.join(sorted(errmodels))}")

print("\n" + ("ALL CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED"))
sys.exit(0 if all(ok) else 1)
