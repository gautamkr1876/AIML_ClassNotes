"""
AcmeBank Assist -- Security Lab
Streamlit front-end for the Advanced_Security_and_Production_Readiness notebook.
Run with:  streamlit run webapp/app.py
"""
import copy
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend as be  # noqa: E402

# Load a .env sitting next to the repo (does not clobber an already-exported var).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except Exception:
    pass

st.set_page_config(page_title="AcmeBank Assist -- Security Lab", page_icon="🏦", layout="wide")


# ============================================================================
# Session state
# ============================================================================
def _init_state():
    ss = st.session_state
    ss.setdefault("audit_log", [])
    ss.setdefault("chat_msgs", [])
    ss.setdefault("chat_history", [])
    ss.setdefault("current_user", "u_alice")
    ss.setdefault("scorecard", None)
    ss.setdefault("redteam_rows", None)
    ss.setdefault("redteam_asr", None)
    ss.setdefault("gdpr_customers", None)
    ss.setdefault("gdpr_audit", None)
    ss.setdefault("chat_draft", "")


_init_state()


# ============================================================================
# Cached, expensive resources -- built once per process, not per rerun.
# ============================================================================
@st.cache_resource(show_spinner="Loading prompt-injection classifier (first run downloads ~450MB)...")
def get_classifier():
    return be.build_injection_classifier()


@st.cache_resource(show_spinner="Building NeMo Guardrails rails...")
def get_nemo_rails(model: str, api_key: str):
    return be.build_nemo_rails(model=model, groq_api_key=api_key)


@st.cache_resource(show_spinner="Building Guardrails AI output guard...")
def get_output_guard():
    return be.build_output_guard()


@st.cache_resource(show_spinner="Loading Presidio PII engine...")
def get_presidio():
    return be.build_presidio()


def build_hooks(use_layer1, use_nemo, use_rbac, use_dlp, model, api_key):
    """Compose the same hook dict shapes the notebook builds in Sections 4-8."""
    hooks = {}
    if use_nemo:
        clf = get_classifier()
        l1 = be.make_input_rail_layer1(clf)
        rails = get_nemo_rails(model, api_key)
        hooks["input_rail"] = be.make_input_rail_nemo(l1, rails)
    elif use_layer1:
        clf = get_classifier()
        hooks["input_rail"] = be.make_input_rail_layer1(clf)
    if use_rbac:
        hooks["tool_rail"] = be.rbac_tool_rail
    if use_dlp:
        guard = get_output_guard()
        analyzer, anonymizer = get_presidio()
        gv = be.make_guardrails_validate_output(guard)
        scrub = be.make_scrub_pii(analyzer, anonymizer)
        hooks["output_rail"] = be.make_output_rail_dlp(gv, scrub)
    return hooks


def safe_call(fn, *args, **kwargs):
    """Run fn, turning API/model errors into a message instead of a crash."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.title("🏦 AcmeBank Assist")
    st.caption("Advanced Security & Production Readiness -- interactive lab")

    api_key_input = st.text_input(
        "GROQ_API_KEY", value=os.environ.get("GROQ_API_KEY", ""), type="password",
        help="Free key at console.groq.com/keys. Kept only in this process's memory.")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    api_key = os.environ.get("GROQ_API_KEY", "")

    model = st.text_input(
        "Model", value=be.DEFAULT_MODEL,
        help="litellm model string. Groq's free-tier catalog changes over time -- "
             "if this 404s, list what's live at console.groq.com and swap it here.")

    st.divider()
    st.subheader("Logged in as")
    current_user = st.selectbox(
        "auth_user_id", be.ALL_LOGIN_USERS,
        index=be.ALL_LOGIN_USERS.index(st.session_state.current_user))
    if current_user != st.session_state.current_user:
        st.session_state.current_user = current_user
        st.session_state.chat_msgs = []
        st.session_state.chat_history = []
    role = be.USER_ROLES.get(current_user, "customer")
    tenant = be.CUSTOMERS.get(current_user, {}).get("tenant") or be.STAFF_TENANTS.get(current_user, "?")
    st.caption(f"role = **{role}**  ·  tenant = **{tenant}**")

    st.divider()
    st.subheader("Defense layers")
    use_layer1 = st.checkbox("① Classifier + heuristics", value=False,
                             help="Section 4: DeBERTa injection classifier + regex/Base64/unicode heuristics.")
    use_nemo = st.checkbox("② NeMo Guardrails self-check", value=False,
                           help="Section 5: LLM-reasoned input rail. Runs Layer 1 first (cheap), NeMo as backstop.")
    use_rbac = st.checkbox("③ RBAC + tenant isolation", value=False,
                           help="Section 8: deterministic permission check on every tool call.")
    use_dlp = st.checkbox("④ Output DLP (Guardrails AI + Presidio)", value=False,
                          help="Sections 6-7: redact secrets/cross-tenant data, then scrub PII.")
    active = [n for n, v in [("Layer1", use_layer1), ("NeMo", use_nemo), ("RBAC", use_rbac), ("DLP", use_dlp)] if v]
    st.caption("Active: " + (", ".join(active) if active else "*none -- undefended agent*"))

    st.divider()
    if st.button("🔄 Reset session (clear chat + audit log)"):
        st.session_state.chat_msgs = []
        st.session_state.chat_history = []
        st.session_state.audit_log = []
        st.rerun()

if not api_key:
    st.warning("Paste a **GROQ\\_API\\_KEY** in the sidebar to use the agent "
               "(free at [console.groq.com/keys](https://console.groq.com/keys)).")

st.title("AcmeBank Assist -- Security Lab")
with st.expander("What is this?", expanded=False):
    st.markdown(
        "An interactive front-end for the notebook's target agent, **AcmeBank Assist**: a "
        "banking support bot with a secret in its system prompt (a canary key), four real "
        "tools (`get_balance`, `get_transactions`, `get_kb_article`, `transfer_funds`), and "
        "multi-tenant customer data. Toggle the four defense layers in the sidebar, chat with "
        "the agent, or run the same 10-attack red-team campaign the notebook uses to measure "
        "Attack Success Rate (ASR) at each layer.")

tab_chat, tab_redteam, tab_scorecard, tab_rbac, tab_compliance = st.tabs(
    ["💬 Chat", "🎯 Red Team", "📊 Defense Scorecard", "🔐 RBAC Tester", "📜 Compliance & Audit"])


# ============================================================================
# Tab: Chat
# ============================================================================
with tab_chat:
    st.caption(
        f"Chatting as **{current_user}**. Pick a canned attack payload below to try it "
        "interactively, or write your own message.")

    attack_labels = ["-- write your own --"] + [f"{a['id']} -- {a['tech']}" for a in be.ATTACKS]
    pick = st.selectbox("Prefill with an attack payload", attack_labels, key="attack_pick")
    if pick != "-- write your own --":
        attack_id = pick.split(" -- ")[0]
        attack = next(a for a in be.ATTACKS if a["id"] == attack_id)
        if attack.get("crescendo"):
            st.info("This is a multi-turn (crescendo) attack -- send each line below as its own "
                     "message, in order, to reproduce it:\n\n" +
                     "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(attack["crescendo"])))
            draft = attack["crescendo"][0]
        else:
            draft = attack["payload"]
    else:
        draft = ""

    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("blocked_by"):
                st.caption(f"🛑 blocked by **{m['blocked_by']}**")
            if m.get("calls"):
                for c in m["calls"]:
                    icon = "✅" if c.get("executed", True) else "🚫"
                    extra = f" -- {c.get('denied_reason','')}" if not c.get("executed", True) else ""
                    st.caption(f"{icon} tool `{c['name']}`({c['args']}){extra}")

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_area("Message", value=draft, height=90, key="chat_text")
        sent = st.form_submit_button("Send", disabled=not api_key)

    if sent and user_msg.strip():
        hooks = build_hooks(use_layer1, use_nemo, use_rbac, use_dlp, model, api_key)
        ctx = be.new_ctx(current_user, audit_log=st.session_state.audit_log)
        st.session_state.chat_msgs.append({"role": "user", "content": user_msg})
        with st.spinner("Agent is thinking..."):
            result, err = safe_call(be.run_agent, user_msg, ctx,
                                    history=st.session_state.chat_history,
                                    hooks=hooks, model=model)
        if err:
            st.error(err)
        else:
            st.session_state.chat_msgs.append({
                "role": "assistant", "content": result["output"],
                "calls": result["calls"], "blocked_by": result.get("blocked_by")})
            if not result.get("blocked_by"):
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                st.session_state.chat_history.append({"role": "assistant", "content": result["output"]})
        st.rerun()


# ============================================================================
# Tab: Red Team
# ============================================================================
with tab_redteam:
    st.caption("Run the notebook's 10-attack catalog against the CURRENT sidebar defense config, "
               "attacking as **u_alice** (matching the notebook's attacker identity).")

    with st.expander("Attack catalog", expanded=False):
        st.dataframe(pd.DataFrame([{"id": a["id"], "OWASP": a["owasp"], "technique": a["tech"]}
                                   for a in be.ATTACKS]), hide_index=True, width="stretch")

    col1, col2 = st.columns([1, 1])
    with col1:
        run_all = st.button("▶ Run all 10 attacks", disabled=not api_key, type="primary")
    with col2:
        single_pick = st.selectbox("...or run one attack", [a["id"] for a in be.ATTACKS], key="rt_single")
        run_one = st.button("▶ Run this one", disabled=not api_key)

    if run_all:
        hooks = build_hooks(use_layer1, use_nemo, use_rbac, use_dlp, model, api_key)
        prog = st.progress(0.0, text="Starting campaign...")
        log = []

        def _cb(i, n, attack, ok):
            prog.progress(i / n, text=f"{i}/{n} -- {attack['id']} {attack['tech']}: "
                                      f"{'WON (defense failed)' if ok else 'stopped'}")

        (rows, asr), err = safe_call(be.run_campaign, hooks=hooks, model=model,
                                     audit_log=st.session_state.audit_log, progress_cb=_cb)
        prog.empty()
        if err:
            st.error(err)
        else:
            st.session_state.redteam_rows = rows
            st.session_state.redteam_asr = asr

    if run_one:
        hooks = build_hooks(use_layer1, use_nemo, use_rbac, use_dlp, model, api_key)
        attack = next(a for a in be.ATTACKS if a["id"] == single_pick)
        with st.spinner(f"Running {single_pick}..."):
            (res, won), err = safe_call(be.run_attack, attack, hooks=hooks, model=model,
                                        audit_log=st.session_state.audit_log)
        if err:
            st.error(err)
        else:
            st.markdown(f"**{attack['id']} -- {attack['tech']}**  ({attack['owasp']})")
            st.write("**Result:**", "🔓 WON (defense failed)" if won else "🛡️ stopped")
            if res.get("blocked_by"):
                st.write("**Blocked by:**", res["blocked_by"])
                st.json(res.get("why") or {})
            st.write("**Agent output:**")
            st.code(res["output"] or "(empty)")
            if res.get("calls"):
                st.write("**Tool calls:**", res["calls"])

    if st.session_state.redteam_rows is not None:
        st.divider()
        asr = st.session_state.redteam_asr
        st.metric("Attack Success Rate", f"{asr:.0%}", help="Lower is better. 0% = fully defended.")
        df = pd.DataFrame(st.session_state.redteam_rows)
        df["result"] = df["attack_won"].map({True: "🔓 WON (defense failed)", False: "🛡️ stopped"})
        df["blocked_by"] = df["blocked_by"].fillna("(RBAC/output rail or refused)")
        st.dataframe(df[["id", "tech", "owasp", "result", "blocked_by", "preview"]],
                    hide_index=True, width="stretch")


# ============================================================================
# Tab: Defense Scorecard
# ============================================================================
with tab_scorecard:
    st.caption(
        "Reproduces Section 9 of the notebook: run the same 10 attacks against four PRESET "
        "configurations -- Undefended, +Layer1, +NeMo, +RBAC & DLP (full stack) -- and watch "
        "ASR fall. This ignores the sidebar toggles and always uses these four presets.")
    st.warning("This is 40 attack runs total (4 configs x 10 attacks) against Groq's free tier. "
               "Budget a minute or two.", icon="⏱️")

    if st.button("▶ Run full 4-layer scorecard", disabled=not api_key, type="primary"):
        configs = [("Undefended", {}), ("+ Layer1 (detect)", None), ("+ NeMo Guardrails", None),
                  ("+ RBAC + DLP (all)", None)]
        try:
            configs[1] = ("+ Layer1 (detect)", build_hooks(True, False, False, False, model, api_key))
            configs[2] = ("+ NeMo Guardrails", build_hooks(True, True, False, False, model, api_key))
            configs[3] = ("+ RBAC + DLP (all)", build_hooks(True, True, True, True, model, api_key))
        except Exception as e:
            st.error(f"Failed to build a defense layer: {type(e).__name__}: {e}")
            configs = None

        if configs:
            prog = st.progress(0.0, text="Starting...")
            total_steps = len(configs) * len(be.ATTACKS)
            done = [0]
            results = {}
            per_attack_full = None
            had_error = False
            for label, hooks in configs:
                def _cb(i, n, attack, ok, label=label):
                    done[0] += 1
                    prog.progress(done[0] / total_steps,
                                  text=f"[{label}] {i}/{n} -- {attack['id']}: "
                                       f"{'WON' if ok else 'stopped'}")
                (rows, asr), err = safe_call(be.run_campaign, hooks=hooks, model=model,
                                             audit_log=[], progress_cb=_cb)
                if err:
                    st.error(f"[{label}] {err}")
                    had_error = True
                    break
                results[label] = asr
                if label == "+ RBAC + DLP (all)":
                    per_attack_full = rows
            prog.empty()
            if not had_error:
                st.session_state.scorecard = {"results": results, "per_attack_full": per_attack_full}

    if st.session_state.scorecard:
        results = st.session_state.scorecard["results"]
        df = pd.DataFrame({"layer": list(results.keys()), "asr": list(results.values())})
        st.bar_chart(df.set_index("layer")["asr"], horizontal=True)
        cols = st.columns(len(results))
        for col, (label, asr) in zip(cols, results.items()):
            col.metric(label, f"{asr:.0%}")

        per_attack = st.session_state.scorecard.get("per_attack_full")
        if per_attack:
            st.subheader("Per-attack breakdown (full stack)")
            df2 = pd.DataFrame(per_attack)
            df2["status"] = df2.apply(
                lambda r: "🔓 WON" if r["attack_won"] else f"🛡️ stopped by {r['blocked_by'] or 'RBAC/output'}",
                axis=1)
            st.dataframe(df2[["id", "tech", "status"]], hide_index=True, width="stretch")
            st.caption(
                "Notice: A07 (cross-tenant) and A08 (transfer) are stopped by **RBAC**, not by any "
                "prompt-based rail -- they're authorization failures, not injection.")


# ============================================================================
# Tab: RBAC Tester
# ============================================================================
with tab_rbac:
    st.caption("RBAC is pure code -- no LLM involved, deterministic, unit-testable. "
               "This is Section 8 of the notebook.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Role -> permissions")
        st.dataframe(pd.DataFrame([{"role": r, "permissions": ", ".join(sorted(p))}
                                   for r, p in be.ROLE_PERMISSIONS.items()]),
                    hide_index=True, width="stretch")
    with c2:
        st.subheader("User -> role / tenant")
        rows = []
        for u in be.ALL_LOGIN_USERS:
            t = be.CUSTOMERS.get(u, {}).get("tenant") or be.STAFF_TENANTS.get(u, "?")
            rows.append({"user": u, "role": be.USER_ROLES.get(u, "customer"), "tenant": t})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Canned test suite (from the notebook)")
    test_rows = []
    for user, tool, args in be.RBAC_TEST_CASES:
        ok, why = be.rbac_tool_rail(tool, args, be.new_ctx(user))
        test_rows.append({"user": user, "tool": tool, "args": str(args),
                          "result": "✅ ALLOW" if ok else "⛔ DENY", "reason": why})
    st.dataframe(pd.DataFrame(test_rows), hide_index=True, width="stretch")

    st.subheader("Interactive tester")
    tc1, tc2, tc3 = st.columns(3)
    t_user = tc1.selectbox("As user", be.ALL_LOGIN_USERS, key="rbac_user")
    t_tool = tc2.selectbox("Tool", list(be.TOOL_IMPLS.keys()), key="rbac_tool")
    with tc3:
        if t_tool in ("get_balance", "get_transactions"):
            t_args = {"user_id": st.text_input("user_id", value="u_alice", key="rbac_arg1")}
        elif t_tool == "get_kb_article":
            t_args = {"topic": st.text_input("topic", value="fees", key="rbac_arg1")}
        else:
            fu = st.text_input("from_user", value="u_alice", key="rbac_arg1")
            to = st.text_input("to_account", value="EXT-999", key="rbac_arg2")
            amt = st.text_input("amount", value="100", key="rbac_arg3")
            t_args = {"from_user": fu, "to_account": to, "amount": amt}
    ok, why = be.rbac_tool_rail(t_tool, t_args, be.new_ctx(t_user))
    st.write("**Result:**", "✅ ALLOW" if ok else f"⛔ DENY -- {why}")


# ============================================================================
# Tab: Compliance & Audit
# ============================================================================
with tab_compliance:
    st.caption("Section 10 of the notebook: GDPR + SOC 2, backed by real (if tiny) operations.")

    st.subheader("Session audit log")
    st.caption("Every tool call this Streamlit session made (from Chat, Red Team, or the RBAC tester's "
               "canned suite goes through `rbac_tool_rail` directly and isn't logged here).")
    if st.session_state.audit_log:
        adf = pd.DataFrame(st.session_state.audit_log)
        adf["allowed"] = adf["allowed"].map({True: "✅ ALLOW", False: "⛔ DENY"})
        st.dataframe(adf, hide_index=True, width="stretch")
    else:
        st.info("Empty -- chat with the agent or run a red-team attack to populate it.")
    if st.button("Clear audit log"):
        st.session_state.audit_log = []
        st.rerun()

    st.divider()
    st.subheader("GDPR: right to erasure (sandboxed demo)")
    st.caption("Runs against a private COPY of the customer data, not the shared demo dataset -- "
               "so clicking this doesn't delete a customer for every other visitor.")
    if st.session_state.gdpr_customers is None:
        st.session_state.gdpr_customers = copy.deepcopy(be.CUSTOMERS)
        st.session_state.gdpr_audit = copy.deepcopy(st.session_state.audit_log)
    erase_target = st.selectbox("Erase user", list(st.session_state.gdpr_customers.keys()) or ["(none left)"])
    ec1, ec2 = st.columns(2)
    if ec1.button("Erase (GDPR Art. 17)") and erase_target in st.session_state.gdpr_customers:
        result = be.gdpr_erase_demo(erase_target, st.session_state.gdpr_customers, st.session_state.gdpr_audit)
        st.success(f"Erased `{erase_target}`: {result}")
    if ec2.button("Reset sandbox copy"):
        st.session_state.gdpr_customers = copy.deepcopy(be.CUSTOMERS)
        st.session_state.gdpr_audit = copy.deepcopy(st.session_state.audit_log)
        st.rerun()
    st.caption(f"Sandbox currently has {len(st.session_state.gdpr_customers)} of "
               f"{len(be.CUSTOMERS)} customer records.")

    st.subheader("Data minimization: support-safe view")
    min_user = st.selectbox("View minimized record for", be.ALL_LOGIN_USERS, key="min_user")
    st.json(be.minimized_view(min_user))

    st.divider()
    st.subheader("Compliance control matrix")
    st.dataframe(pd.DataFrame(be.CONTROL_MATRIX, columns=["Control", "OWASP LLM Top-10", "Compliance mapping"]),
                hide_index=True, width="stretch")
