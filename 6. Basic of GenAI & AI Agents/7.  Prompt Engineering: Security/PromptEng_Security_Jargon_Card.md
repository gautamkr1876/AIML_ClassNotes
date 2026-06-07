<a id="top"></a>
# Prompt Engineering: Security — Jargon Card

> **Use this file like a dictionary.** Skim it once (~6 min) before opening the notebook. Then keep it open in a side tab — when you hit an unknown word while reading, look it up here in 20 seconds instead of Googling for 5 minutes.
>
> **Companion:** read [`PromptEng_Security_Reading_Brief.md`](./PromptEng_Security_Reading_Brief.md) FIRST. This card is just the dictionary.
> **The notebook:** `L7(Jun 6th)_ Prompt Engineering_ Security.ipynb` in this folder.

---

## A

**Anonymization** — Replacing sensitive data in text with non-identifiable stand-ins *before* it leaves your systems. Presidio's `AnonymizerEngine` does this — e.g., turning `212-555-5555` into `<PHONE_NUMBER>`, a mask, or a fake number. The reverse is **de-anonymization**.

**Adversarial prompt** — A prompt deliberately crafted to make a model misbehave (leak data, produce harmful content, ignore its rules). The raw material of **red teaming**.

## B

**Base64 (attack enhancement)** — An encoding scheme that turns text into an opaque string of letters/digits. Attackers Base64-encode a malicious request and ask the model to "decode and execute it," hoping to slip past filters that only scan plain text. One of the three enhancement techniques the notebook demos.

**BENIGN** — One of the three labels Meta's **Prompt-Guard-86M** classifier assigns to text. Means "safe, normal input" (the other two are INJECTION and JAILBREAK).

## C

**CyberSecEval** — A security benchmark tool for LLMs, part of Meta's **Purple Llama** framework (the same family as Llama Guard). Mentioned for context; the notebook focuses on Llama Guard.

**Context-aware recognition** — A Presidio feature that *boosts* a match's confidence only when expected surrounding words are present. A bare 10-digit number scores low; the same number near "customer with ID" scores high. Implemented with `LemmaContextAwareEnhancer`. Cuts **false positives**.

## D

**Defense in depth** — The security principle of stacking *multiple* independent safeguards so no single failure is catastrophic. The notebook stresses that Llama Guard (or any one tool) is "merely a layer," never the whole defense.

**De-anonymization (round-trip)** — Restoring the original PII values in the model's *response* after the model only ever saw redacted placeholders. Production-grade requirement; the notebook shows a simplified placeholder→original mapping and notes real systems use Presidio's `DeanonymizeEngine`.

**DeBERTa** — The transformer architecture underneath Prompt-Guard-86M. You don't need its internals — just know it's a small *classifier* model (not a generative one), which is why it runs on CPU.

**DeepTeam** — An open-source framework (built on **DeepEval**) that automates red teaming: it generates attacks, enhances them, runs them against your model, and scores responses across 50+ vulnerability types. The notebook's "scaling up" tool, contrasted with hand-written attacks.

**Denial of Wallet (DoW)** — An attack where an adversary deliberately drives up your *costs* on a pay-per-use LLM service by triggering excessive usage. A flavor of **Unbounded Consumption** (LLM10).

**Direct injection** — A **prompt injection** where the attacker types override instructions straight into the chat box ("Ignore previous instructions…"). Real example: a car-dealership chatbot talked into offering a car for $1. Contrast with **indirect injection**.

**Dual-gate** — The Llama Guard architecture of checking content *both* at the **input gate** (before the LLM) and the **output gate** (after the LLM). Catches malicious input and unsafe output.

## E

**Embedding** — A numeric vector representing a piece of text's meaning, used by RAG to find relevant documents. **Vector and Embedding Weaknesses** (LLM08) covers attacks on these, like embedding-inversion (reverse-engineering the source text from the vector).

**EntityRecognizer** — Presidio's base class for a *code-based* custom PII detector. You subclass it and write `analyze()` logic when a simple regex won't capture your PII (e.g., detecting internal corporate URLs). Contrast with `PatternRecognizer` (regex-based).

**Excessive Agency (LLM06:2025)** — An OWASP risk: giving an agent more power than its task needs. Three root causes — excessive **functionality** (extra tools), excessive **permissions** (over-broad credentials), excessive **autonomy** (acts without confirmation). Fix: least privilege + human-in-the-loop.

## F

**False positive** — Flagging something as sensitive/unsafe when it isn't (e.g., redacting a dollar amount that happened to be 10 digits). The notebook's context-aware recognition exists specifically to reduce these.

**Faker** — A Python library that generates realistic fake data (names, emails, etc.). Presidio can use it as an anonymization operator — replacing a real email with a plausible fake one instead of a `<TAG>`.

**Firewall (LLM firewalling)** — Treating a safety classifier like a network firewall: it inspects traffic *into* and *out of* the LLM and blocks what's unsafe. The mental frame for Llama Guard's dual gates.

## G

**Guardrail** — A rule or filter that checks an LLM's inputs/outputs and blocks policy violations. Llama Guard's safety categories (O1–O7) are guardrails; the notebook shows adding a *custom* one (electronic-communication abuse / phishing).

## H

**Hallucination** — When a model generates confident but false content, often by filling knowledge gaps. The root cause of **Misinformation** (LLM09). Real consequences cited: Air Canada's chatbot gave wrong refund advice (airline lost the lawsuit); ChatGPT invented fake legal cases an attorney then cited in court.

**Human-in-the-loop** — Requiring a person to approve high-impact actions before an agent executes them. The recommended mitigation for **Excessive Agency** (e.g., confirm before deleting files).

## I

**Improper Output Handling (LLM05:2025)** — An OWASP risk: downstream systems trusting LLM output without checking it, letting attackers run SQL, shell, or JavaScript that the model was tricked into producing. Fix: **zero-trust** the output — validate, sanitize, filter before use.

**Indirect injection** — A **prompt injection** hidden inside content the system *processes* — a resume, a webpage, an email, or retrieved **RAG** context — so the user never types anything suspicious. Especially dangerous in RAG: an attacker just plants instructions in an indexable source and waits for the retriever to pull them in.

**INJECTION** — A label from Prompt-Guard-86M: text trying to hijack the LLM's behavior (e.g., poisoned RAG context). Distinct from JAILBREAK (which targets safety filters).

**Input gate** — In Llama Guard, the check applied to the *user prompt before* it reaches the target LLM. Mitigates prompt injection and sensitive-info disclosure, and avoids paying API costs on content that'll be blocked anyway.

## J

**Jailbreak** — An attack that breaks the **model's** safety filters (via roleplay, encoding, or emotional manipulation) to make it produce content it was trained to refuse. Contrast with **prompt injection**, which breaks your *application's* logic. They overlap but aren't identical — an injection that makes your bot recommend a competitor is an exploit, not a safety violation.

**JAILBREAK** — The Prompt-Guard-86M label for text trying to override safety filters (e.g., "ignore previous instructions").

## L

**LangChain** — A framework that provides an abstraction layer to talk to different LLMs. In the Llama Guard pipeline it's the glue between the guard gates and the target LLM (OpenAI).

**Least privilege (principle of)** — Grant an agent only the *minimum* functionality and permissions its task requires. The core fix for **Excessive Agency**.

**Llama Guard** — Meta's experimental **Llama2-7b** safety *classifier* model that labels a conversation `safe` or `unsafe` (plus the violated category). Used as a **dual-gate** firewall. Part of **Purple Llama**. Needs a GPU; can be customized with new safety categories. The notebook: "not sufficient on its own."

**LlamaGuard vs Prompt-Guard** — Two different Meta safety tools. **LlamaGuard** (7B, GPU) checks for *content* harm (violence, hate, etc.). **Prompt-Guard-86M** (tiny, CPU) specifically detects *injection/jailbreak* attempts. The notebook notes LlamaGuard caught only 2 of 7 injection attacks that Prompt-Guard catches.

**LLM01–LLM10** — The OWASP Top 10 for LLMs (2025), each risk tagged with an ID. See the Reading Brief for the full list. ⚠️ One section of the notebook (Llama Guard) uses *older 2023* numbering where LLM02 = Insecure Output Handling and LLM06 = Sensitive Info Disclosure — different from the 2025 list.

## M

**MART (Multi-Round Automatic Red Teaming)** — Meta's automated red-teaming method. Cited stat: an **84.7% reduction** in violation rates after just four adversarial training rounds — proof that feeding red-team findings back into training improves safety.

**Misinformation (LLM09:2025)** — An OWASP risk: the model generating false but convincing content, usually via **hallucination**. Fixes: RAG (ground in verified sources) + human fact-checking for high-stakes domains.

**Model weakness** — A flaw traced to *training* (biased data → biased output; PII overfit → leakage; weak alignment → easy jailbreak). Fixed by retraining, RLHF, or adversarial fine-tuning. Contrast with **system weakness**.

**Multi-turn escalation** — An attack that starts innocuous and gradually shifts the conversation toward a harmful goal over 5–10 messages, evading filters that only inspect a single message.

## O

**OperatorConfig** — Presidio's way to choose *how* to anonymize each entity type: replace with a tag, **mask** with a character, substitute a **Faker** fake, or apply a custom lambda. Different entities can use different operators in one call.

**Output gate** — In Llama Guard, the check applied to the *LLM's response before* it returns to the user. Mitigates improper output handling and sensitive-info disclosure.

**OWASP** — The **Open Worldwide Application Security Project**, a nonprofit (since 2001) behind widely used security risk lists. Its **Gen AI Security Project** (launched May 2023) produces the **OWASP Top 10 for LLMs**.

## P

**PII (Personally Identifiable Information)** — Data that identifies a person: name, email, phone, SSN, credit card, passport, etc. Leaking it breaks privacy laws like **GDPR**. The notebook uses **Presidio** to detect and redact it before LLM calls.

**Presidio** — Microsoft's open-source Python framework to **detect and anonymize PII**. Two components: `AnalyzerEngine` (finds PII — type, span, confidence) and `AnonymizerEngine` (redacts it). Extensible with custom recognizers; uses **spaCy** internally.

**Prompt-Guard-86M** — Meta's lightweight (86M-parameter, CPU-only, DeBERTa-based) classifier that labels text BENIGN / INJECTION / JAILBREAK. A focused injection detector, complementary to the broader Llama Guard.

**Prompt injection (LLM01:2025)** — The top OWASP LLM risk: user input that manipulates the model's behavior in unintended ways, exploiting the fact that the trusted **system prompt** and untrusted **user input** are concatenated into one text stream. Two types: **direct** and **indirect**.

**Purple Llama** — Meta's umbrella framework for "open trust and safety" in generative AI. Includes **Llama Guard** and **CyberSecEval**.

## R

**RAG (Retrieval-Augmented Generation)** — Grounding LLM answers in documents pulled from a knowledge base via **embeddings**. Reappears here as both a *defense* (against misinformation) and an *attack surface* (poisoned knowledge base → **indirect injection**; **LLM08** embedding weaknesses).

**Red teaming** — Deliberately attacking your *own* LLM with adversarial prompts to find vulnerabilities (bias, leakage, harmful content, injection) before real attackers do. Borrowed from military "red team vs blue team" exercises.

**Refusal phrases** — The keyword list ("I can't", "I'm sorry", "against my guidelines"…) the notebook's simple scorer searches for to decide whether a model *refused* an attack (→ PASS).

**ROT13** — A trivial cipher that shifts each letter 13 places. Used as an attack enhancement: encode the malicious request so keyword filters don't see it, betting the model decodes it anyway.

**RRT (Recognize, Reduce, Trap)** — A design method for mitigating prompt injection. Llama Guard operates in the **Trap** phase — catching malicious content the design couldn't prevent.

## S

**Sanitization** — Cleaning data to remove dangerous content: stripping PII from inputs, or stripping executable code (SQL/HTML/JS) from outputs before downstream systems use them.

**SBOM (Software Bill of Materials)** — An inventory of every third-party component (models, datasets, libraries) in your system, so you can scan them for tampering. A mitigation for **Supply Chain** risk (LLM03).

**Sensitive Information Disclosure (LLM02:2025)** — An OWASP risk: the model unintentionally exposing PII, proprietary algorithms, or confidential business data. Fixes: data sanitization, input/output validation, restrict data sources.

**spaCy** — An NLP library Presidio uses internally for tokenization and named-entity recognition. The notebook downloads its `en_core_web_lg` model.

**Supply Chain (LLM03:2025)** — An OWASP risk: vulnerabilities in third-party datasets, pre-trained models, or fine-tuning components (e.g., the PoisonGPT attack — a tampered model uploaded to a public repo). Fix: vet sources, maintain an **SBOM**, run custom evaluations.

**System prompt** — The hidden developer-written instructions that set the model's role and rules, concatenated with user input. **System Prompt Leakage** (LLM07) is the risk of these instructions (and any secrets in them) being exposed.

**System Prompt Leakage (LLM07:2025)** — An OWASP risk: hidden instructions revealing sensitive info — credentials, internal logic (loan-approval rules), or user roles. Fix: never put secrets in system prompts; store them externally.

**System weakness** — A flaw in the *application layer*, not the model: insecure API endpoints, overpowered tool access, weak prompt templates, missing input validation. Fixed architecturally (sandboxing, least privilege, guardrails). Contrast with **model weakness**.

## T

**Threat model** — A structured map of what your specific app does, what could go wrong, and which risks actually matter for it. The notebook's #1 red-teaming best practice: "start with your threat model, not the tool."

## U

**Unbounded Consumption (LLM10:2025)** — An OWASP risk: uncontrolled request volume or input size overloading the system or inflating costs (including **Denial of Wallet**). Fix: rate limits, input-size caps, monitoring.

**unsafe (Llama Guard verdict)** — Llama Guard returns `safe` or `unsafe`; an unsafe verdict is followed by the violated category code (e.g., `unsafe\nO1` = Violence & Hate).

**UNSAFE_INDICATORS** — In the notebook's red-team scorer, the regex patterns (shell commands, `<script>`, step-by-step how-tos, PII patterns) that mark a response as a **FAIL** if found.

## V

**Vector and Embedding Weaknesses (LLM08:2025)** — An OWASP risk: attacks on the RAG embedding layer — embedding-inversion (recovering source data from vectors) and data poisoning of the vector store. Fix: access controls, isolation, validation pipelines.

[🔝 Back to top](#top)
