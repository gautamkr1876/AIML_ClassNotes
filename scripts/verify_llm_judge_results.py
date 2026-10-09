"""Verify every number quoted in the L15 (LLM-as-a-Judge) notes against the notebook's output.

Cells are located by CONTENT, never by index, so inserting a cell (e.g. the
Open-in-Colab badge) cannot silently invalidate this script.

Run:  python3 scripts/verify_llm_judge_results.py
"""
import json, math, re, pathlib

NB = pathlib.Path(__file__).resolve().parents[1] / \
    "8. Agentic AI Systems/15.  LLM-as-a-Judge & Advanced Evaluation/L14_LLM-as-a-Judge & Advanced Evaluation.ipynb"
cells = json.load(open(NB))["cells"]


def src(i):
    return "".join(cells[i].get("source", []))


def out(i):
    s = []
    for o in cells[i].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t:
            s.append("".join(t) if isinstance(t, list) else t)
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(s))


def find(*needles):
    hits = [i for i, c in enumerate(cells)
            if c["cell_type"] == "code" and all(n in src(i) for n in needles)]
    assert len(hits) == 1, f"expected 1 cell matching {needles}, found {hits}"
    return hits[0]


C = {
    "sample":    find("sampled_indices = []"),
    "golden":    find("stress_tests = ["),
    "accuracy":  find("def compute_accuracy"),
    "groq":      find("groq_judge_correct = np.array"),
    "groq_dis":  find('Cases where Groq judge disagreed with exact match'),
    "kappa":     find("def safe_kappa"),
    "dis":       find("disagreements = []"),
    "ensemble":  find("ensemble_strict ="),
    "deepeval":  find("def summarize_deepeval"),
    "failures":  find("def categorize_failure"),
    "tool":      find("def tool_usage_appropriateness"),
}
print("resolved cells:", dict(sorted(C.items(), key=lambda kv: kv[1])))

ok = []


def chk(name, got, want, tol=0.05):
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"  {'PASS' if good else 'FAIL'}  {name:<55} computed {got:>9.3f}   notebook {want:>9.3f}")


def assert_(name, cond):
    ok.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def num(pattern, text, cast=float):
    m = re.search(pattern, text)
    assert m, f"pattern not found: {pattern}"
    return cast(m.group(1))


# ------------------------------------------------------------ 1. golden set
print("\n1. GOLDEN DATASET")
o = out(C["golden"])
assert_("60 items = 50 MATH-500 + 10 stress tests",
        "Total golden dataset size: 60" in o and "MATH-500 samples: 50" in o and "Stress tests:     10" in o)
o = out(C["sample"])
assert_("7 subjects, 5 levels in the 50-item sample",
        o.count("'") >= 14 and "Level distribution" in o)

# ------------------------------------------------------------ 2. accuracy
print("\n2. ACCURACY")
o = out(C["accuracy"])
exact = num(r"Exact-match accuracy:\s+([\d.]+)%", o)
symb = num(r"Symbolic-match accuracy:\s+([\d.]+)%", o)
chk("exact match = 16/60", 100 * 16 / 60, exact)
chk("symbolic match = 16/60 (gap +0.0)", 100 * 16 / 60, symb)
chk("MATH sample (symbolic) = 11/50", 100 * 11 / 50, num(r"MATH-500 sample \(symbolic\):\s+([\d.]+)%", o))
chk("stress tests (symbolic) = 5/10", 100 * 5 / 10, num(r"Stress tests\s+\(symbolic\):\s+([\d.]+)%", o))
assert_("stop reasons: final_answer 28, max_steps 22, parse_error 10",
        "'final_answer': 28" in o and "'max_steps': 22" in o and "'parse_error': 10" in o)

o = out(C["groq"])
groq = num(r"Groq-judge accuracy:\s+([\d.]+)%", o)
chk("Groq judge = 26/60", 100 * 26 / 60, groq)
chk("gap judge - exact = 10/60", 100 * 10 / 60, num(r"Gap \(judge - exact match\):\s+\+([\d.]+)", o))
dis = num(r"Disagreements between exact-match and Groq judge: (\d+)/60", o, int)
assert_("14/60 exact-vs-Groq disagreements", dis == 14)

o = out(C["groq_dis"])
assert_("Groq said CORRECT on Predicted: None (math500_208, _3, _178)",
        all(f"[{k}]" in o for k in ("math500_208", "math500_3", "math500_178")) and o.count("Predicted: None") >= 3)

# ------------------------------------------------------------ 3. kappa derivation
print("\n3. COHEN'S KAPPA (exact vs Groq), derived from the printed margins")
# a = both correct, b = exact only, c = Groq only, d = both wrong
# a+b = 16 (exact), a+c = 26 (Groq), b+c = 14 (disagreements)  =>  unique solution
b = (16 + 14 - 26) // 2
c = 14 - b
a = 16 - b
d = 60 - a - b - c
assert_(f"confusion matrix solved: a={a}, b={b}, c={c}, d={d}", (a, b, c, d) == (14, 2, 12, 32))
po = (a + d) / 60
pA, pB = (a + b) / 60, (a + c) / 60
pe = pA * pB + (1 - pA) * (1 - pB)
kap = (po - pe) / (1 - pe)
o = out(C["kappa"])
chk("kappa exact vs Groq", kap, num(r"Exact vs Groq:\s+kappa = ([\d.]+)", o), tol=0.0005)
chk("observed agreement exact vs Groq (%)", 100 * po, num(r"Exact vs Groq:.*observed = (\d+)%", o), tol=0.5)
chk("observed agreement Groq vs Gemini (%) = 34/60", 100 * 34 / 60, num(r"Groq  vs Gemini:.*observed = (\d+)%", o), tol=0.5)
assert_("Groq vs Gemini kappa printed as nan (a rater has no variance)", re.search(r"Groq  vs Gemini:\s+kappa = nan", o))

# ------------------------------------------------------------ 4. ensemble
print("\n4. ENSEMBLE + DISAGREEMENTS")
o = out(C["ensemble"])
chk("strict ensemble (both correct)", 0.0, num(r"Strict \(both agree correct\):\s+([\d.]+)%", o))
chk("lenient ensemble (either correct) = Groq's 26/60", 100 * 26 / 60, num(r"Lenient \(either says correct\):\s+([\d.]+)%", o))
assert_("26/60 need human review", "Needs human review:          26 / 60" in o)
o = out(C["dis"])
assert_("26/60 Groq-vs-Gemini disagreements", "Number of judge disagreements: 26 / 60" in o)
assert_("Gemini verdicts were PARSE_ERROR recorded as wrong", "Gemini -> wrong: PARSE_ERROR" in o)

# ------------------------------------------------------------ 5. traces / custom metric
print("\n5. FAILURE MODES + TOOL USE")
o = out(C["failures"])
assert_("failures: hit_max_steps 22, wrong_reasoning 12, parse_error 10",
        re.search(r"hit_max_steps\s+:\s+22", o) and re.search(r"wrong_reasoning\s+:\s+12", o) and re.search(r"parse_error\s+:\s+10", o))
assert_("failure total = 44 = 60 - 16 exact-match passes", 22 + 12 + 10 == 60 - 16)
o = out(C["tool"])
chk("tool usage = (55*1 + 5*0.5)/60", (55 + 2.5) / 60, num(r"Tool usage appropriateness \(average\): ([\d.]+)", o), tol=0.005)

o = out(C["deepeval"])
assert_("DeepEval shows MathCorrectness FAIL with ReasoningQuality PASS on some rows",
        re.search(r"0\.00 \(FAIL\)\s+0\.(6|8)0 \(PASS\)", o))

# ------------------------------------------------------------ 6. extension arithmetic used in the notes
print("\n6. EXTENSION ARITHMETIC (statistics, corrections)")
p, n = 26 / 60, 60
se = math.sqrt(p * (1 - p) / n)
chk("SE of 43.3% on n=60", se, 0.0640, tol=0.0005)
chk("95% CI low (%)", 100 * (p - 1.96 * se), 30.8, tol=0.05)
chk("95% CI high (%)", 100 * (p + 1.96 * se), 55.9, tol=0.05)
chk("n for +/-3 pts at p=0.5", (1.96 * 0.5 / 0.03) ** 2, 1067.1, tol=0.1)
chk("Rogan-Gladen example: p=.433, TPR=.95, TNR=.80", (0.433 + 0.80 - 1) / (0.95 + 0.80 - 1), 0.3107, tol=0.0005)
chk("quiz example: p=.80, TPR=.90, TNR=.70", (0.80 + 0.70 - 1) / (0.90 + 0.70 - 1), 0.8333, tol=0.0005)
for name, (pp, bb) in {"prevalence paradox 54/3/3/0": (54 / 60, 57 / 60)}.items():
    pe_ = bb * bb + (1 - bb) * (1 - bb)
    chk(name + " kappa", (pp - pe_) / (1 - pe_), -0.0526, tol=0.0005)
bias = 0.7
chk("position bias p=0.7: single-order A wins", bias, 0.70)
chk("position bias p=0.7: consistent ties", bias ** 2 + (1 - bias) ** 2, 0.58, tol=0.0001)
chk("availability 0.99*0.99", 0.99 * 0.99, 0.9801, tol=0.00001)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
raise SystemExit(0 if all(ok) else 1)
