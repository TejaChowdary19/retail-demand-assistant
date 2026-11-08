import streamlit as st
from src import service as sv


def product_picker(label="Product"):
    prods = sv.products()
    cur = st.session_state.get("_product", prods[0])
    pid = st.selectbox(label, prods, index=prods.index(cur))
    st.session_state["_product"] = pid
    return pid
