# AcmeBank Assist -- Security Lab

A Streamlit front-end for `Advanced_Security_and_Production_Readiness.ipynb`.
The notebook's agent, 10-attack red-team catalog, and four defense layers
(injection classifier + heuristics, NeMo Guardrails, RBAC, Guardrails AI +
Presidio DLP) are ported into `backend.py`; `app.py` is the UI on top.

## Setup

```bash
cd webapp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Needs Python 3.10 (see the version note in `requirements.txt` if you're on a
different version).

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). Paste a
free [Groq API key](https://console.groq.com/keys) into the sidebar, or drop
it in a `.env` file (`GROQ_API_KEY=...`) at the repo root or in `webapp/` --
the app loads it automatically without ever writing it to disk itself.

## What's in each tab

- **Chat** -- talk to the agent as any of the demo users, with the sidebar's
  four defense toggles applied live. A dropdown prefills any of the 10 attack
  payloads into the message box.
- **Red Team** -- run the full 10-attack catalog (or one attack) against the
  CURRENT sidebar defense config and see which attacks won vs. were stopped,
  and by what.
- **Defense Scorecard** -- reproduces the notebook's Section 9: runs the same
  10 attacks against four PRESET configs (Undefended -> +Layer1 -> +NeMo ->
  +RBAC & DLP) and charts ASR falling to 0%. This is 40 live model calls, so
  it takes a minute or two.
- **RBAC Tester** -- the permission model is plain, deterministic code (no
  LLM). Browse the role/user tables, the notebook's canned test suite, or
  try your own (user, tool, args) combination.
- **Compliance & Audit** -- the session's audit log, a control-matrix mapping
  each defense to OWASP LLM Top-10 / SOC 2 / GDPR, and a GDPR right-to-erasure
  demo that runs against a private sandbox copy of the customer data (not the
  shared demo dataset).

## Notes on how this differs from the notebook

- **Model**: defaults to `groq/qwen/qwen3.8-27b` (editable in the sidebar).
  Groq's free-tier catalog changes over time -- if it 404s, list what's live
  (`curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"`)
  and swap the model string.
- **Per-session state**: the audit log and chat history are scoped to your
  Streamlit session (`st.session_state`), not shared globals like the
  notebook's module-level `AUDIT_LOG` -- so multiple browser tabs against the
  same running app don't see each other's chat/audit trail. The customer/KB
  data itself is still shared, read-mostly demo data across all sessions in
  one running process, same as the notebook.
- **DLP redaction fix**: `backend.py` composes the two Guardrails AI output
  validators (secret-leak, cross-tenant) as two chained single-validator
  Guards instead of `Guard().use(A).use(B)`. In guardrails-ai 0.11.0 that
  chaining has a real bug -- if the first validator fails and redacts, but
  the second then *passes* on the (already-redacted) text, the whole result
  silently reverts to the original, un-redacted output. The notebook's own
  demo cell never surfaces this because its sample text always trips both
  validators at once; a canary-only leak would slip through unredacted. See
  the docstring on `build_output_guard()` for details.
