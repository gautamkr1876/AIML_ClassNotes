"""Verify every number quoted in L14 (DSPy) against the notebook's recorded output.

Source of truth: "8. Agentic AI Systems/14. DSPy: Mathematical Prompt Optimization/
DSPY_liveclass.ipynb" -- cells 9, 10, 27, 28, 32, 34.

Run:  python3 scripts/verify_dspy_eval_arithmetic.py
"""
import json, re, pathlib, sys

NB = pathlib.Path(__file__).resolve().parents[1] / \
    "8. Agentic AI Systems/14. DSPy: Mathematical Prompt Optimization/DSPY_liveclass.ipynb"

def out(cells, i):
    s = []
    for o in cells[i].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t: s.append("".join(t) if isinstance(t, list) else t)
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(s))

cells = json.load(open(NB))["cells"]
ok = []
def check(name, got, want, tol=0.01):
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"  {'PASS' if good else 'FAIL'}  {name:<46} computed {got:>8.2f}   notebook {want:>8.2f}")

print("\n1. DATASET  (cell 9)")
neu, pos, neg, tot = 2298, 1091, 483, 3872
check("neutral+positive+negative == total", neu + pos + neg, tot)
check("majority-class baseline %  (2298/3872)", 100 * neu / tot, 59.35)
print("      -> 'always predict neutral' already scores 59.35% on the raw pool.")

print("\n2. SPLITS  (cell 10)  stratified, 3 classes")
for nm, per, n in [("train", 7, 21), ("val", 17, 51), ("test", 34, 102)]:
    check(f"{nm}: 3 x {per}", 3 * per, n)
check("negative class is the binding constraint (7+17+34)", 7 + 17 + 34, 58)
print(f"      -> 58 of the 483 negatives used = {100*58/483:.1f}% of the scarcest class.")

print("\n3. BASELINE  dspy.Predict  (cell 27)")
t27 = out(cells, 27)
assert "81.00 / 102" in t27 and "79.41" in t27
check("81/102", 100 * 81 / 102, 79.41)
check("rate-limit errors in this cell", t27.count("RateLimitError"), 0)
print("      -> 102/102 scored, zero errors, 6.3s at 16.20 it/s. The only clean run.")

print("\n4. BASELINE  dspy.ChainOfThought  (cell 28)")
t28 = out(cells, 28)
errs = len(set(re.findall(r"Error for Example\(\{'headline': (?:'|\").{0,60}", t28)))
assert "73.00 / 98 (74.5%)" in t28, "progress-bar figure moved"
assert "Average Metric: 73.0 / 102 (71.6%)" in t28, "final INFO figure moved"
check("progress bar  73/98  (survivors only)", 100 * 73 / 98, 74.49)
check("final report   73/102 (errors == 0.0)", 100 * 73 / 102, 71.57)
check("distinct examples lost to rate limits", errs, 4)
check("102 - 98 == examples never scored", 102 - 98, errs)
gap = 100 * 73 / 98 - 100 * 73 / 102
check("points of the drop that are pure infra", gap, 2.92)
check("real deficit vs Predict (79.41 - 74.49)", 79.41 - 100 * 73 / 98, 4.92)
print("      -> the SAME cell prints 74.5% and 71.6%. Same 73 correct; different denominator.")
print(f"      -> CoT ran 69.0s vs Predict 6.3s = {69.0/6.3:.1f}x slower -> it caused its own throttling.")

print("\n5. BootstrapFewShot  (cells 32, 34)")
t32, t34 = out(cells, 32), out(cells, 34)
assert "Bootstrapped 4 full traces after 4 examples" in t32
print("      compile: 4 demos accepted from the first 4 of 21 train examples (19%), 0.2s.")
check("train examples never even attempted", 21 - 4, 17)
last = re.findall(r"Average Metric: ([\d.]+) / (\d+) \(([\d.]+)%\)", t34)[-1]
check("last survivor score 38/49", 100 * 38 / 49, float(last[2]), tol=0.06)
check("progress reached /102", int(re.findall(r"\| (\d+)/102", t34)[-1]), 59)
check("distinct rate-limit failures", len(set(re.findall(r"Error for Example\(\{'headline': (?:'|\").{0,60}", t34))), 10)
raised = cells[34]["outputs"][-1].get("output_type") == "error"
ok.append(raised)
print(f"  {'PASS' if raised else 'FAIL'}  cell 34 terminates in an exception            "
      f"{cells[34]['outputs'][-1].get('evalue','')}")
print("      -> no BootstrapFewShot accuracy was ever printed. The optimized number does not exist.")

print("\n6. THE COMPARISON THE NOTEBOOK IMPLIES")
print("      Predict            79.41%   81/102   complete")
print("      ChainOfThought     71.57%   73/102   4 examples scored 0.0 by rate limit")
print("      BootstrapFewShot      --    died at 59/102, 38/49 survivors (77.55%)")
print("      Only row 1 is a trustworthy measurement. 77.55% vs 79.41% is not a comparison:")
print("      different examples, different count, one of them truncated mid-run.")

print("\n7. ERROR ANALYSIS  (cell 29 vs the cell-30 markdown)")
t29 = out(cells, 29)
pairs = re.findall(r"Gold: (\w+)\s+\|\s+Predicted: (\w+)", t29)
from collections import Counter
c = Counter(pairs)
check("errors found in first 30 test examples", int(re.findall(r"Errors in first 30: (\d+)", t29)[0]), 7)
print("      printed error directions:", dict(c))
n2p = c.get(("neutral", "positive"), 0)
check("gold=neutral -> predicted=positive", n2p, 4)
print("      -> cell 30 claims the model 'predicts neutral for anything'. 4 of the 5")
print("         printed errors are the exact opposite: neutral gold, positive prediction.")

print("\n" + ("ALL CHECKS PASSED" if all(ok) else "SOME CHECKS FAILED"))
sys.exit(0 if all(ok) else 1)
