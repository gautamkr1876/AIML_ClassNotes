"""Verify the BM25 derivation in the L11/L12 notes.

The notebook calls rank_bm25's BM25Okapi and prints 4.501 / 1.553 / 1.553 without
showing the arithmetic. This script re-implements Okapi BM25 from the published
formula and checks that it reproduces those numbers exactly, so the derivation in
the notes is verified rather than asserted.

Run:  python3 scripts/verify_bm25_derivation.py
"""
import json, re, math, os

NB = ("8. Agentic AI Systems/12.  Hybrid Search & Advanced Retrieval/"
      "L11_and_L12_Hybrid_Search_&_Advanced_Retrieval.ipynb")
EXPECTED = [("D06", 4.501), ("D07", 1.553), ("D08", 1.553)]
K1, B, EPS = 1.5, 0.75, 0.25          # rank_bm25 BM25Okapi defaults


def load_corpus(path):
    nb = json.load(open(path))
    src = next(s for s in ("".join(c.get("source", [])) for c in nb["cells"])
               if s.startswith("# Extended corpus for today"))
    ns = {}
    exec(src.split("df = pd.DataFrame")[0], ns)
    docs = ns["DOCUMENTS"]
    return [d[0] for d in docs], ["{}. {}".format(d[2], d[3]) for d in docs]


def tokenize(text):
    return re.findall(r"[a-z0-9\-]+", text.lower())


def build(tokenised):
    n = len(tokenised)
    df = {}
    for d in tokenised:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    idf, neg, total = {}, [], 0.0
    for w, f in df.items():
        v = math.log(n - f + 0.5) - math.log(f + 0.5)
        idf[w] = v
        total += v
        if v < 0:
            neg.append(w)
    for w in neg:                       # rank_bm25 floors negative idf
        idf[w] = EPS * (total / len(idf))
    return df, idf, sum(len(d) for d in tokenised) / n


def score(doc, q, idf, avgdl):
    dl, total, parts = len(doc), 0.0, []
    for t in q:
        tf = doc.count(t)
        if tf == 0:
            continue
        num = idf.get(t, 0.0) * tf * (K1 + 1)
        den = tf + K1 * (1 - B + B * dl / avgdl)
        parts.append((t, tf, idf.get(t, 0.0), num / den))
        total += num / den
    return total, parts, dl


def main():
    ids, texts = load_corpus(NB)
    tok = [tokenize(t) for t in texts]
    df, idf, avgdl = build(tok)
    q = tokenize("INC-2847 postmortem")

    scored = sorted(((ids[i], score(tok[i], q, idf, avgdl)[0])
                     for i in range(len(tok))), key=lambda p: -p[1])
    print(f"N = {len(tok)}   avgdl = {avgdl:.2f}   k1 = {K1}   b = {B}")
    print("query tokens:", q, "\n")
    ok = True
    for (gid, gs), (eid, es) in zip(scored[:3], EXPECTED):
        good = gid == eid and abs(gs - es) < 0.001
        ok &= good
        print(f"  {gid}  {gs:.3f}   expected {eid} {es}   {'OK' if good else 'MISMATCH'}")

    total, parts, dl = score(tok[ids.index("D06")], q, idf, avgdl)
    print(f"\nD06 breakdown  (dl = {dl} tokens)")
    for t, tf, i_, c in parts:
        print(f"  {t:<12} df={df[t]:>2}  tf={tf}  idf={i_:+.4f}  -> {c:.4f}")
    print(f"  {'TOTAL':<12} {total:.4f}")
    print("\nVERIFIED" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
