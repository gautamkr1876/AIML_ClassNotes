<a id="top"></a>
# LLM Evaluation Jargon Card

> **Use this file like a dictionary.** Skim it once (~7 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading the notebook, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`LLM_Evals_Reading_Brief.md`](./LLM_Evals_Reading_Brief.md) FIRST. This card is just the dictionary.

---

## A

**Accuracy** — Fraction of predictions that match the ground-truth label. `(TP + TN) / (TP + TN + FP + FN)`. Standard classification metric, but misleading on imbalanced data — that's why precision/recall/F1 exist.

**Agentic system** — An LLM-powered system that takes actions (calls tools, retrieves data, iterates) rather than just answering. Evaluating one means scoring not just final answers but also tool-use correctness.

**Answer Relevance** — An LLM-based metric that scores whether the model's answer actually addresses the user's question (vs. drifting off topic). Implemented in Opik as a built-in metric.

## B

**BERT-base** — The base BERT model (12 layers, 768-dim embeddings). The default backbone for BERTScore.

**BERTScore** — A semantic similarity metric. Encodes both candidate and reference with BERT (contextual embeddings), computes pairwise cosine similarities between every token pair, then aggregates via **greedy matching** into precision/recall/F1. Catches paraphrases that BLEU/ROUGE miss.

**BERTScore F1** — Harmonic mean of BERTScore precision and recall. The "one number to report."

**BERTScore Precision** — *"How well does each candidate token find a match in the reference?"* For each candidate token, find its maximum cosine similarity with any reference token; average.

**BERTScore Recall** — *"How well does each reference token find a match in the candidate?"* For each reference token, find its maximum cosine similarity with any candidate token; average.

**Bigram** — A sequence of two consecutive tokens. "the cat" is a bigram. ROUGE-2 uses bigrams.

**BLEU (Bilingual Evaluation Understudy)** — A metric for translation quality. Measures **n-gram precision**: what fraction of the candidate's 1-, 2-, 3-, 4-grams also appear in the reference. Combined as a geometric mean. Higher is better.

## C

**Candidate** — The model's generated text being evaluated.

**Contextual embedding** — A vector representation of a token that **depends on its surrounding context**. Same word in different sentences gets different embeddings (e.g., "bank" in "river bank" vs. "money bank"). Produced by BERT, RoBERTa, GPT, etc. The whole point of BERTScore.

**Cosine similarity** — Distance metric for embeddings: `cos(θ) = A · B / (‖A‖‖B‖)`. Range `[-1, 1]`. `1` = identical direction. The default for comparing text/image embeddings.

**Criteria-based evaluation** — An LLM-as-judge mode where the judge scores a response on a **specific dimension** (e.g., tone, factual accuracy, clarity). Output is a numeric score with a rubric. Contrast with **pairwise comparison**.

## D

**Document frequency (DF)** — How many documents in a corpus contain a given token. Used in IDF: rare-across-the-corpus tokens carry more information.

**Domain expert** — A subject-matter specialist (lawyer, doctor, support rep) whose judgment is the ground truth for what "good" looks like. Must be involved when defining LLM-judge criteria.

## E

**Eval / Evaluation** — Systematic measurement of an LLM's outputs. Distinct from "testing" (which assumes deterministic outputs) — LLM evals must handle non-determinism, multiple valid answers, and subjective quality.

**Evaluation prompt** — The prompt fed to the *judge* LLM (not the system-under-test). Tells the judge what to score and how. Building a good one is most of the work in LLM-as-a-judge.

## F

**F1 score** — Harmonic mean of precision and recall: `2 · P · R / (P + R)`. Used both in classification and in BERTScore.

**Faithfulness** — A metric: *"Are the facts in the answer actually supported by the retrieved context?"* High = no hallucination. (Defined in Ragas — close cousin of "groundedness" elsewhere.)

**False negative (FN)** — A positive example that the model wrongly predicted as negative. Hurts recall.

**False positive (FP)** — A negative example that the model wrongly predicted as positive. Hurts precision.

## G

**Greedy matching** — In BERTScore: for each candidate token, find its best (highest-cosine-similarity) match in the reference, no constraints on uniqueness. Cheap (O(n·m)) and good enough. Alternative would be optimal-assignment matching (Hungarian algorithm), which is much more expensive.

**Groundedness** — Same idea as **Faithfulness**: are the answer's claims grounded in the retrieved context?

## H

**Hallucination** — When the model produces content that sounds confident but isn't supported by the input/context. The #1 problem in LLM products. The Opik **Hallucination** metric tries to detect this automatically.

**Hyperparameter** — A model/evaluation configuration setting (temperature, max_tokens, top_p). Not learned during training.

## I

**IDF (Inverse Document Frequency)** — A weight: `IDF(t) = log(N / DF(t))` where `N` = total docs, `DF(t)` = number of docs containing token `t`. **Rare tokens get high weight; common tokens like "the" get low weight.** Used in BERTScore to emphasise informative words.

**Inference** — Running a trained model to produce outputs. Different from training.

**Instance / Sample** — One row of an evaluation dataset (e.g., one (question, expected answer, context) tuple).

## L

**LCS (Longest Common Subsequence)** — The longest sequence of tokens that appears in both candidate and reference **in the same order** (but not necessarily contiguous). ROUGE-L uses LCS.

**LLM-as-a-Judge (a.k.a. LLM-as-Critic)** — Using an LLM (often GPT-4o or Claude Sonnet) to **evaluate** another LLM's outputs. Works because **critiquing is easier than creating** — the judge only has to recognise quality, not produce it. Standard practice for evaluating open-ended generation.

## M

**Model eval** — Evaluating the **model itself** on standard benchmarks (MMLU, HumanEval, GSM8K, etc.). Tells you "is GPT-4o better than Claude 3.5 Sonnet at math?". Useful for picking which model to use.

**Moderation** — An LLM-based metric that detects harmful, toxic, or inappropriate content. Implemented in Opik.

## N

**N-gram** — A sequence of `n` consecutive tokens. Unigram = 1 token, bigram = 2, trigram = 3, 4-gram = 4. BLEU averages over 1-, 2-, 3-, 4-grams.

**Non-determinism** — In LLMs: same input + same model → different outputs across calls (due to sampling). Makes evaluation fundamentally harder than for classifiers.

## O

**Offline evaluation** — Running evaluation on a **fixed dataset** in a lab, before shipping. Cheap, repeatable, but doesn't see real users. BLEU/ROUGE/BERTScore/LLM-as-judge are all offline.

**Online evaluation** — Measuring performance with **real users in production**. User engagement, latency, business KPIs (click-through rate, conversion, retention). Catches problems offline tests can't.

**Open-ended generation** — A task with multiple valid outputs: translation, summarisation, dialogue. The hard case for evals — no single "correct" answer.

**Opik** — An open-source LLM evaluation platform from Comet. Provides metrics (BLEU, ROUGE, BERTScore, LLM-based ones), experiment tracking, and a dashboard. The notebook's primary tool.

## P

**Pairwise comparison** — An LLM-as-judge mode where the judge sees **two responses** to the same prompt (A and B) and picks the better one (or a tie). Used for ranking models or prompt variants head-to-head. Research shows GPT-4 pairwise judgments agree ~80%+ with crowd-sourced human preferences.

**Polysemous** — A word with multiple meanings depending on context. "Bank" (financial vs. river). Contextual embeddings handle this; static embeddings (word2vec) can't.

**Porter stemmer** — A classic algorithm for reducing words to their root form ("running" → "run"). Used by ROUGE when "use_stemmer=True" to handle morphological variants.

**Precision** — `TP / (TP + FP)`. Fraction of predicted positives that are correct. "If the model says yes, how often is it right?"

**Product eval** — Evaluating the **whole application** (model + prompts + retrieval + business logic) on real user inputs. Tells you "does our customer-support bot actually answer customer questions well?" — which is what matters for shipping.

**Prompt injection** — An adversarial attack where the user's input contains hidden instructions ("ignore previous instructions and...") that hijack the system prompt. Tested by tools like `garak`.

## R

**RAG (Retrieval-Augmented Generation)** — A pattern: retrieve relevant docs first, then have the LLM answer using those docs as context. Evaluating a RAG system means scoring both retrieval quality and answer quality.

**Recall** — `TP / (TP + FN)`. Fraction of true positives that the model found. "Of all the real positives, how many did the model catch?"

**Reference** — The gold-standard or human-written answer that the model's output is compared against. (Some metrics like LLM-as-judge don't need a reference.)

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** — A family of metrics for summarisation. Measures **n-gram recall**: what fraction of the reference's tokens appear in the candidate. Counterpart to BLEU's precision focus.

**ROUGE-1** — Unigram (single-token) overlap recall.

**ROUGE-2** — Bigram overlap recall.

**ROUGE-L** — Longest Common Subsequence-based recall. Captures word order without requiring exact contiguity.

**ROUGE-Lsum** — ROUGE-L computed per sentence and then aggregated (for multi-sentence summaries).

## S

**Semantic similarity** — How close two texts are in **meaning**, not surface form. "Cats are cute" and "Felines are adorable" have high semantic similarity but zero word overlap. Captured by embedding-based metrics like BERTScore.

**Sentence transformers** — Models specifically trained to produce sentence-level embeddings where semantically similar sentences have similar vectors. Used widely for retrieval and similarity.

**SST-2** — Stanford Sentiment Treebank v2, a sentiment classification dataset (binary positive/negative). The notebook uses 100 examples for sentiment evaluation.

**SQuAD** — Stanford Question Answering Dataset. The notebook uses 100 examples for QA evaluation.

**Stemming** — Reducing inflected words to a common root ("running", "ran", "runs" → "run"). Lets metrics treat morphological variants as the same token.

**Subjective quality** — Attributes that don't have an objective "right answer": tone, style, helpfulness, clarity. LLM-as-judge is the best automated way to score these.

## T

**Temperature** — Sampling parameter (0 = deterministic, > 0 = stochastic). For evaluation: set to **0** so results are reproducible.

**Token** — Atomic unit of text (subword, word, or punctuation). All n-gram metrics work on tokens after tokenization.

**Top-k / Top-p** — Sampling strategies. Set carefully during evaluation so results are reproducible.

**Trigram** — Three consecutive tokens. Used in BLEU's averaging.

**True negative (TN)** — A negative example correctly predicted as negative.

**True positive (TP)** — A positive example correctly predicted as positive.

## U

**Unigram** — A single token. ROUGE-1 uses unigrams.

## V

**Variance (in eval)** — How much the same input produces different outputs across runs. High variance makes single-shot metrics unreliable — average over multiple runs.

---

[🔝 Back to top](#top)
