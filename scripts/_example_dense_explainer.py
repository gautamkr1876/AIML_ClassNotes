import sys; sys.path.insert(0, "scripts")
from diagram_kit import *
set_output("8. Agentic AI Systems/12.  Hybrid Search & Advanced Retrieval/images")

fig, ax = canvas(11.4, 7.6)
ax.add_patch(Rectangle((0, 0), 100, 100, fc=WHITE, ec="none", zorder=0))
frame(ax)
header(ax, 2, "Sparse Retrieval with BM25",
       "Rank by how RARE a matched term is, not how often it appears.",
       glyph="[ tokens ] -> [ score ]")

# ---------------- left rail ----------------
railcard(ax, 3.5, 61, 27, 16, "blue", "?", "What it is",
         ["Matches literal tokens, not meaning.",
          "One slot per vocab word, almost all zero."])
railcard(ax, 3.5, 41, 27, 18, "green", "!", "Why it matters",
         ["Users paste IDs, codes, acronyms.",
          "Exact strings are what dense loses.",
          "Cheap, CPU-only, no training."])
panel(ax, 3.5, 17, 27, 20, "Scoring Formula")
ax.text(17, 32.8, "idf(t) ~ log( N / df(t) )", fontsize=9.6, color=NAVY,
        ha="center", va="center", fontweight="bold", fontfamily=MONO, zorder=5)
for i, (lbl, val) in enumerate([("N", "total documents = 20"),
                                ("df(t)", "docs containing t"),
                                ("rare t", "-> large weight"),
                                ("common t", "-> ~ zero")]):
    yy = 28.0 - i * 2.9
    ax.text(6.5, yy, lbl, fontsize=7.6, color=BLUE, va="center",
            fontweight="bold", fontfamily=MONO, zorder=5)
    ax.text(14.5, yy, val, fontsize=7.6, color=BODY, va="center",
            fontfamily=SANS, zorder=5)

# ---------------- right: code + result ----------------
panel(ax, 33, 17, 63.5, 60)

codeblock(ax, 35.5, 50, 28, 22, step=1, step_tint="blue", title_text="Build the index", fs=7.0, code="""from rank_bm25 import BM25Okapi

def tokenize(text):
    # keep the hyphen!
    return re.findall(
        r"[a-z0-9\\-]+", text.lower())

DOCS = [tokenize(t) for t in DOC_TEXTS]
BM25 = BM25Okapi(DOCS)""")

codeblock(ax, 66, 50, 28, 22, step=2, step_tint="green", title_text="Score a query", fs=7.0, code="""def bm25_search(query, k=5):
    scores = BM25.get_scores(
        tokenize(query))
    order = np.argsort(-scores)[:k]
    return [(DOC_IDS[i],
             DOC_TITLES[i],
             float(scores[i]))
            for i in order]""")

# result bars
stepbadge(ax, 37.7, 45.8, 3, "amber")
ax.text(40.5, 45.8, 'Result   query = "INC-2847 postmortem"', fontsize=8.8,
        color=NAVY, va="center", fontweight="bold", fontfamily=SANS, zorder=5)
res = [("INC-2847 Postmortem", "D06", 4.501, 1), ("INC-3102 Postmortem", "D07", 1.553, 0),
       ("INC-2915 Postmortem", "D08", 1.553, 0), ("Guardrail Overview", "D01", 0.000, 0)]
for i, (t, d, s, win) in enumerate(res):
    yy = 40.8 - i * 5.0
    card(ax, 35.5, yy - 1.9, 58.5, 4.2, tint=None if win else "grey",
         accent="blue" if win else None, z=4)
    ax.text(38, yy, f"{t}  ({d})", fontsize=8.0, color=INK if win else BODY,
            va="center", fontweight="bold" if win else "normal",
            fontfamily=SANS, zorder=6)
    bw = 22 * (s / 4.501)
    if bw > 0.4:
        ax.add_patch(FancyBboxPatch((66, yy - 1.0), bw, 2.0,
                     boxstyle="round,pad=0,rounding_size=0.5",
                     fc=BLUE if win else "#C7D2E4", ec="none", zorder=6))
    ax.text(92.5, yy, f"{s:.3f}", fontsize=8.2, color=INK if win else FAINT,
            ha="right", va="center", fontfamily=MONO,
            fontweight="bold" if win else "normal", zorder=6)

# failure strip
card(ax, 35.5, 17.8, 58.5, 5.9, tint="rose", z=4)
ax.text(38, 21.9, "Fails on paraphrase", fontsize=8.2, color=ROSE, va="center",
        fontweight="bold", fontfamily=SANS, zorder=6)
ax.text(38, 19.3, '"stop attackers tricking our chatbot"  ->  ranks Access Request Procedure 3rd',
        fontsize=7.6, color=ROSE, va="center", fontfamily=SANS, zorder=6)

takeaway(ax, "INC-2847 appears in 1 of 20 documents, so its IDF weight dominates: "
             "4.501 against 1.553 for the runner-up.",
         y=2.5, h=11.5,
         right_lines=["Rare term -> high IDF.", "Keep the hyphen.",
                      "No concept of synonyms."])
save(fig, "03_bm25_worked_example.png")
