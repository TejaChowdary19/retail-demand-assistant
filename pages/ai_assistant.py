import os
import pandas as pd
import streamlit as st
from src import service as sv, llm_tools as lt
from src.ui import product_picker

MAX_USER_MESSAGES = 15
art = sv.artifacts()


def api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


st.title("AI Assistant")
pid = product_picker("Selected product (context for the assistant)")
saved = st.session_state.get("_saved_inputs", {})
scn = saved.get(pid)
st.caption(f"Context: **{pid}** | planner inputs: **{'your saved scenario' if scn else 'demo defaults'}** "
           f"| data: historical demo, forecasts start {art['forecast_start']}")
if scn:
    st.json(scn, expanded=False)

with st.expander("Get the same numbers without the assistant"):
    st.caption("These use the same functions the assistant's tools call.")
    h = st.slider("At-risk horizon (days)", 1, 28, 7)
    risk = sv.get_products_at_risk(h, saved)
    st.write(f"{risk['n_at_risk']} product(s) may run out within {h} days (stock on order excluded).")
    if risk["at_risk"]:
        st.dataframe(pd.DataFrame(risk["at_risk"]), hide_index=True, width="stretch")
    st.json(sv.get_inventory_recommendation(pid, scn), expanded=False)

EX = ["Which products may run out within 7 days?", f"What is the forecast for {pid}?",
      f"How much should I order for {pid} if delivery takes 10 days?",
      "Which five products need the most replenishment?", f"How well did the model perform for {pid}?"]
cols = st.columns(len(EX))
for c, q in zip(cols, EX):
    if c.button(q, key=f"ex_{q}"):
        st.session_state["_pending"] = q

chat = st.session_state.setdefault("chat", [])
for m in chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            with st.expander("Underlying tool results"):
                st.json(m["trace"])

prompt = st.chat_input("Ask about forecasts, inventory or model performance") or st.session_state.pop("_pending", None)
if prompt:
    n_user = sum(m["role"] == "user" for m in chat)
    key = api_key()
    if n_user >= MAX_USER_MESSAGES:
        st.warning(f"Session limit of {MAX_USER_MESSAGES} questions reached. Reload the page to start over.")
    elif not key:
        st.error("The assistant is not configured (no API key). The Forecast Explorer and Inventory Planner still work, "
                 "and the panel above shows the same numbers.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        history = [{"role": m["role"], "content": m["content"]} for m in chat] + [{"role": "user", "content": prompt}]
        with st.chat_message("assistant"):
            try:
                with st.spinner("Working..."):
                    text, trace = lt.chat(history, key, lt.build_context(pid, scn, art), saved)
                st.markdown(text)
                if trace:
                    with st.expander("Underlying tool results"):
                        st.json(trace)
                chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text, "trace": trace}]
            except Exception as e:
                st.error(f"The assistant is temporarily unavailable ({type(e).__name__}). Forecasts, the inventory "
                         "planner, and the panel above still work and give the same numbers.")
