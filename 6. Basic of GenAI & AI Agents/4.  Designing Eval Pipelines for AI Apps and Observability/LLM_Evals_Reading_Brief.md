<a id="top"></a>
# LLM Evaluation & Observability — Reading Brief

> **Read this ONCE, end to end, before opening the notebook.** Target time: ~22 minutes. By the time you reach the notebook, every word will already make sense — you'll be confirming what you already know, not learning blind.
>
> **Side reference:** keep [`LLM_Evals_Jargon_Card.md`](./LLM_Evals_Jargon_Card.md) open in another tab while reading the notebook. When an unknown word appears, look it up here.
> **The notebook:** `L4_ Designing Eval Pipelines for AI Apps and Observability.ipynb` in this folder.

---

## 🎯 30-second TL;DR

**LLM evaluation is fundamentally harder than ML evaluation.** Classifiers have one right answer (yes/no); LLMs produce **open-ended text** with many valid forms.

The notebook walks through a layered toolkit:

1. **N-gram metrics (BLEU, ROUGE)** — fast and cheap; only count overlapping words. Miss paraphrases.
2. **BERTScore** — uses contextual embeddings to compare **meaning**, not just exact words. Catches paraphrases.
3. **LLM-as-a-Judge** — use a strong LLM (GPT-4o) to score outputs on subjective criteria. The only way to grade tone, helpfulness, etc.
4. **Online evaluation** — once shipped, measure real-user signals (engagement, latency, satisfaction).

All implemented in **Opik** (Comet's LLM eval platform). Headline insight: *you can't ship what you can't measure* — and measuring LLMs takes more than a confusion matrix.

---

## 🗺️ Agenda — what the notebook teaches, in order

1. **Understanding LLM products** — chatbots, code assistants, RAG systems, agents.
2. **Why LLM evals are hard** — four challenges: non-determinism, no single right answer, wide input range, unique risks (hallucination, prompt injection).
3. **Model evals vs product evals** — benchmark a model vs evaluate your application.
4. **Automated evaluation overview** — when manual review breaks down (anything past ~100 examples).
5. **BLEU** — n-gram precision metric (originally for translation).
6. **ROUGE** — n-gram recall family (originally for summarisation): ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum.
7. **BERTScore — the semantic upgrade** — contextual embeddings → cosine similarity → greedy matching → precision/recall/F1.
8. **BERTScore deep dive** — IDF weighting (emphasise rare/informative tokens), strengths, limitations, when to use.
9. **LLM-as-a-Judge — why it works** — critiquing is easier than creating.
10. **Three types of LLM judges** — pairwise, criteria-based, reference-based.
11. **Pairwise comparison** — head-to-head model/prompt selection.
12. **Criteria-based evaluation** — score on tone, clarity, factual accuracy.
13. **5-step process** to build an LLM judge: define scenario → prepare data → label → write prompt → iterate.
14. **LLM-based metrics in Opik** — Answer Relevance, Hallucination, Moderation.
15. **Online evaluation in production** — Gmail Smart Compose, Google Translate, ChatGPT — what they actually measure.

---

## 🧠 The big idea

> **You can't grade an essay with `==`.**

Traditional ML: input → label → check if it matches the gold label. Simple. LLMs: input → free-form text → ??? How do you check if "Cats are wonderful pets" is a correct answer to "Tell me about cats"?

**The eval stack solves this in layers:**

- **Exact match** → useless for free text.
- **N-gram overlap** (BLEU, ROUGE) → works for short, structured outputs; misses paraphrases.
- **Semantic similarity** (BERTScore) → catches paraphrases; misses tone, helpfulness, correctness.
- **LLM-as-a-Judge** → catches everything; expensive; needs careful prompt design.
- **Online metrics** → ground truth from real users; only available after shipping.

You **layer them up**: cheap automated metrics gate every code change; LLM-judge runs on a held-out set; online metrics validate in production. No single metric is enough.

---

## 📖 Core concept primers

### 1. Why LLM evals are hard — four challenges

> **🪜 Mental model:** classical ML eval = grading a multiple-choice test. LLM eval = grading an essay.

The notebook lists four reasons LLMs break traditional testing:

1. **Non-determinism.** Same input → different outputs across calls (sampling). You can't `assertEqual(model(x), expected)`. Workaround: set `temperature=0` *or* sample many times and aggregate.
2. **No single correct answer.** "Write a polite refusal email" has thousands of valid forms. Exact-match metrics give false zero scores.
3. **Wide input range.** A chatbot can be asked anything. You can't cover the input space with a test set. Workaround: stratified sampling + adversarial probes.
4. **Unique risks.** Hallucination, prompt injection, jailbreaks, toxic output, PII leakage. None of these exist in classifiers; all of them need dedicated evals.

**Why it matters in the notebook.** This is the framing for everything that follows. Each metric tackles one or more of these challenges.

### 2. Model evals vs Product evals

> **🪜 Mental model:** **model eval = test the engine on a dyno.** **Product eval = drive the car around with passengers.**

| | Model Evals | Product Evals |
|---|---|---|
| **What you test** | The raw model | The whole app (model + prompts + retrieval + business logic) |
| **Dataset** | Public benchmarks (MMLU, HumanEval, GSM8K) | Your own users' queries + your gold answers |
| **Question answered** | "Is GPT-4o better than Claude 3.5 at math?" | "Does our support bot answer customer questions well?" |
| **Run when** | Picking which model to use | Every code change; ongoing in production |
| **Tools** | LM Evaluation Harness, OpenAI evals | Opik, Ragas, DeepEval — your own eval suite |

**Punchline.** Model evals help you *choose* a model. Product evals tell you whether your *application* works. You need **both**, and product evals are where the real engineering effort goes.

### 3. N-gram metrics (BLEU, ROUGE) — fast and cheap, often wrong

> **🪜 Mental model:** count how many short word-runs match between candidate and reference. Cheap, but blind to meaning.

**BLEU** measures **precision**: of the candidate's 1-, 2-, 3-, 4-grams, what fraction also appear in the reference? Geometric mean of the four precisions (with a length-brevity penalty). Originally for translation.

**ROUGE** measures **recall**: of the reference's tokens, what fraction appear in the candidate? Variants:
- **ROUGE-1**: unigram recall — what fraction of reference words made it into the candidate?
- **ROUGE-2**: bigram recall — adjacent-word pairs.
- **ROUGE-L**: longest common subsequence recall — captures word order without requiring adjacency.
- **ROUGE-Lsum**: ROUGE-L per sentence, then aggregated (for multi-sentence summaries).

**The fatal flaw.** Both ignore meaning. Compare:

- Reference: *"The cat sat on the mat."*
- Candidate A: *"The feline rested on the rug."* → meaning preserved, **BLEU/ROUGE ≈ 0**.
- Candidate B: *"The cat the on mat sat."* → word-salad, **BLEU/ROUGE > 0**.

**Why they're still useful.** Cheap (no GPU), deterministic, well-understood. Good for translation/summarisation regression tests where the reference is highly constrained.

### 4. BERTScore — semantic similarity that catches paraphrases

> **🪜 Mental model:** instead of counting matching words, **embed every token with BERT** and check how close the embeddings are.

**The pipeline:**

1. Encode **candidate** with BERT → vector per token.
2. Encode **reference** with BERT → vector per token.
3. Compute **pairwise cosine similarities** between every candidate token and every reference token → a similarity matrix.
4. **Greedy matching** — for each candidate token, take the max similarity with any reference token (and vice versa).
5. Aggregate into **precision** (avg of candidate-side maxes), **recall** (avg of reference-side maxes), **F1** (harmonic mean).

**Why it catches paraphrases.** "Feline" and "cat" land near each other in BERT's embedding space, so their cosine similarity is high. BLEU/ROUGE see them as different words; BERTScore sees them as the same idea.

**IDF weighting (optional add-on).** Common words like "the" carry less information than rare words like "felidae." Weight each token's contribution by its IDF (`log(N / DF(t))`) to emphasise informative matches.

**Strengths.** Handles paraphrases. Handles word order via the bidirectional matching. Modest extra cost (one BERT forward pass).
**Limitations.** Needs the right BERT model (multilingual, domain-specific) for the data. Doesn't measure factual correctness — only similarity. Embeddings can match wrong concepts ("good" vs "bad" are surprisingly close in some embedding spaces).

### 5. LLM-as-a-Judge — the workhorse for subjective quality

> **🪜 Mental model:** **critiquing is easier than creating.** A model that can't write a great answer can still recognise a great one.

**Why it works.** Coming up with a great essay is hard. Reading two essays and picking the better one is much easier. The same is true for LLMs: GPT-4 may not be perfect at writing helpful answers, but it's pretty good at *scoring* them. Research shows GPT-4 pairwise judgments agree ~80%+ with crowdsourced human preferences.

**Three types of judges:**

| Type | What it does | Use case |
|---|---|---|
| **Pairwise** | "Which is better, A or B (or tie)?" | Comparing two models, two prompts, or two RAG configurations head-to-head |
| **Criteria-based** | "Score this on a 1–5 scale for {tone / clarity / factual accuracy}." | Tracking quality dimensions over time; production monitoring |
| **Reference-based** | "Compare this answer to the gold answer." | When you have a reference and want a similarity-aware score |

**The 5-step process to build a judge** (the notebook's signature workflow):

1. **Define the scenario** — what task? What quality matters?
2. **Prepare data** — collect a representative set of inputs and outputs.
3. **Label with humans** — get gold labels from domain experts (often just yourself).
4. **Write the eval prompt** — explicit rubric, examples (few-shot), output format (often JSON with score + reasoning).
5. **Iterate** — compare judge scores to human labels. Tweak the prompt until agreement is high.

**Why step 3 matters.** If you don't know what "good" looks like, your judge won't either. Skip the human labeling and your judge will produce confident-but-wrong scores. **Domain experts must be involved.**

**Opik built-in LLM-based metrics** (no judge-prompt-writing required):
- **Answer Relevance** — does the answer address the question?
- **Hallucination** — does the answer contain claims not in the context?
- **Moderation** — is the content safe?

### 6. Offline vs Online evaluation

> **🪜 Mental model:** **offline = lab, fixed dataset. Online = production, real users.**

| | Offline | Online |
|---|---|---|
| **Data** | Static eval set | Live user traffic |
| **Speed** | Hours | Days–weeks |
| **Cost** | Cheap | Free (you're already serving) |
| **Catches** | Regressions; algorithmic improvements | UX issues; business impact; rare-input edge cases |
| **Examples** | BLEU / ROUGE / BERTScore / LLM-as-judge on a held-out set | A/B test response styles; track CTR on Gmail Smart Compose suggestions; measure ChatGPT thumbs-up rate |

**Real-world examples from the notebook:**

- **Gmail Smart Compose** — measures acceptance rate (did the user actually use the suggestion?), latency, and abandonment.
- **Google Translate** — measures user re-translations (signals the first translation was wrong), copy-rate.
- **ChatGPT** — measures thumbs-up/down, regenerate rate, conversation length.

**The discipline:** **offline gates code changes; online validates business impact.** Skip offline and you ship regressions. Skip online and you don't actually know if your app works.

---

## 🔥 The headline evaluation stack — at a glance

| Layer | Metric type | When it fires | Catches | Cost |
|---|---|---|---|---|
| **1** | Unit tests / exact match | Every commit | Coding bugs | Free |
| **2** | N-gram metrics (BLEU, ROUGE) | Every PR | Output regressions in constrained tasks | Cheap |
| **3** | Semantic similarity (BERTScore) | Every PR | Paraphrase-aware regressions | Cheap (1 BERT pass) |
| **4** | LLM-as-judge (criteria, pairwise) | Nightly / pre-deploy | Subjective quality issues | $$ (one GPT-4 call per example) |
| **5** | LLM-based metrics (Answer Relevance, Hallucination, Moderation) | Sampled in production | Safety + grounding issues | $$ |
| **6** | Online metrics (engagement, satisfaction, latency) | Always-on | UX and business problems | Free |

**The discipline.** Stack them up. Cheap-and-fast at the bottom (every commit), slow-and-expensive at the top (sampled in production). No single layer is enough.

---

## 🧮 Key metrics & formulas

- **Precision** = `TP / (TP + FP)` — *"if I say yes, am I right?"*
- **Recall** = `TP / (TP + FN)` — *"did I catch all the real positives?"*
- **F1** = `2·P·R / (P+R)` — harmonic mean; one number to report.
- **Accuracy** = `(TP+TN) / (TP+TN+FP+FN)` — misleading on imbalanced data.
- **Cosine similarity** = `A·B / (‖A‖·‖B‖)` — embedding distance for semantic comparison.
- **IDF** = `log(N / DF(t))` — weights rare/informative tokens higher.
- **BLEU** = geometric mean of 1-, 2-, 3-, 4-gram precisions × brevity penalty.
- **ROUGE-L** = LCS-based recall — captures word order without requiring adjacency.

**Reproducibility knobs for eval runs:**
- OpenAI: `temperature=0` (deterministic).
- HuggingFace: `do_sample=False` or `temperature=0.01` (near-deterministic).
- Always set `max_tokens=100` (or whatever your scenario expects) — let `finish_reason="length"` warn you of truncations.

---

## 🗺️ Notebook reading map — where to spend your attention

| Cells | What it teaches | How to read |
|---|---|---|
| **1–13** | LLM products; types of systems (RAG, agents, chains); key questions evals must answer | **Skim** — ~4 min. Framing. |
| **14–18** | Four challenges of LLM evals; product vs model evals | **Read carefully** — ~5 min. The "why this is hard" foundation. |
| **19–45** | Environment setup (Opik); SST-2 sentiment evaluation pipeline; custom Opik metrics (Accuracy / F1 / Precision / Recall) | **Focus on the pattern** — ~6 min. The Opik plumbing is the same for every metric below. |
| **46–62** | SQuAD QA evaluation; BLEU; ROUGE-1 / -2 / -L / -Lsum | **FOCUS** — ~7 min. These are the n-gram baselines. |
| **63–110** | BERTScore — motivation, core idea, token embeddings, greedy matching, P/R/F1, IDF weighting, code walkthrough, paraphrase/word-order experiments | **FOCUS** — ~10 min. The semantic upgrade. Slow down on the greedy-matching diagram. |
| **111–125** | LLM-as-a-Judge — why it works, types (pairwise, criteria), 5-step build process | **FOCUS** — ~8 min. The workhorse for subjective evals. |
| **126–129** | Opik LLM metrics (Answer Relevance, Hallucination, Moderation) — wired into the same eval pipeline | **Read normally** — ~3 min. Out-of-the-box drop-ins. |
| **130–134** | Online evaluation; Gmail/Translate/ChatGPT case studies; offline-vs-online best practices | **Read carefully** — ~5 min. The "real-world bridge" section. |

**Total target read time for the notebook itself:** ~50 min. Add this brief's ~22 min and you're at **~70–75 min**, vs. a cold read of probably 2+ hours given the metric density.

---

## ✅ Walk-away checklist

After the notebook, you should be able to say in your own words:

- [ ] **Four reasons LLM evals are harder than classifier evals** — non-determinism, no single right answer, wide input range, unique risks.
- [ ] **The difference between model evals and product evals** — and why product evals are the ones that matter for shipping.
- [ ] **What BLEU and ROUGE measure** — n-gram precision and recall. And **the one failure mode they share** — both miss paraphrases.
- [ ] **The four steps of BERTScore** — embed candidate, embed reference, cosine-similarity matrix, greedy match → P/R/F1.
- [ ] **Why LLM-as-a-Judge works** — critiquing is easier than creating; GPT-4 ~80% agrees with humans.
- [ ] **Pairwise vs criteria-based judges** — when to use each.
- [ ] **The 5-step process to build an LLM judge** — define / prepare / label / prompt / iterate. The role of domain experts.
- [ ] **The full eval stack** — n-gram → semantic → LLM-judge → LLM-based metrics → online — and what each catches that the layer below misses.

---

## 🎯 5-question self-check

Answer in your head, then check below. **No peeking.**

1. Reference: *"The cat sat on the mat."* Candidate: *"The feline rested on the rug."* What will BLEU/ROUGE say? What will BERTScore say? Why the difference?
2. You're evaluating a customer-support chatbot. You can run only one metric. Which would you pick — accuracy on a labelled QA set, or LLM-as-judge scoring tone + factual accuracy? Why?
3. In BERTScore: what's the difference between **precision** and **recall**? Use the "every candidate token finds its best reference match" / "every reference token finds its best candidate match" framing.
4. Why does the notebook insist that the 5-step LLM-judge process must involve a **domain expert**? What goes wrong if you skip step 3 (human labeling)?
5. You ship a new prompt. Offline BERTScore goes up 5 points. User retention drops 8%. What's the most likely explanation, and what does this tell you about the relationship between offline and online metrics?

---

<details>
<summary><b>Click to reveal answers</b></summary>

1. **BLEU and ROUGE will give near-zero scores** because the candidate shares almost no n-grams with the reference ("the" and "on" maybe). **BERTScore will give a high score** because "feline" and "cat" have near-identical BERT embeddings, as do "rested"/"sat" and "rug"/"mat." The difference: n-gram metrics look at surface form; BERTScore looks at meaning. This example is exactly why semantic metrics exist.

2. **LLM-as-judge** — by far. Accuracy on a labelled QA set only tells you if the bot picks the right facts. It tells you nothing about whether the bot is **polite**, **clear**, or **helpful** — which is what determines whether customers come back. For a customer-support product, subjective quality dominates objective correctness.

3. **Precision** = *"For every candidate token, how well does it find a match in the reference?"* — for each candidate token, take its max cosine similarity with any reference token, then average. Measures "coverage of what the candidate said."
   **Recall** = *"For every reference token, how well does it find a match in the candidate?"* — for each reference token, take its max cosine similarity with any candidate token, then average. Measures "coverage of what the reference said."
   F1 is the harmonic mean. Precision punishes the candidate for adding extra stuff; recall punishes it for missing important stuff.

4. **The judge LLM doesn't know what "good" looks like for *your* domain unless you tell it.** Without human labels, you have no ground truth to compare the judge's scores against — so you can't tell if the judge is producing useful scores or confidently-wrong ones. The risk: you end up with a confident-looking metric that's totally uncorrelated with what users actually value. Domain experts ground the judge in reality.

5. **The new prompt is producing more paraphrased-but-worse answers.** BERTScore measures semantic similarity to a reference — so a more *eloquent* answer that *says the same thing differently* will score higher even if it's confusing, off-tone, or unhelpful to real users. **Offline metrics are necessary but not sufficient.** They catch *some* regressions but can be Goodharted — optimised in ways that don't transfer to user value. That's exactly why the stack needs the **online evaluation** layer on top: real users are the only true ground truth.

</details>

---

[🔝 Back to top](#top) · [→ Jargon Card](./LLM_Evals_Jargon_Card.md)
