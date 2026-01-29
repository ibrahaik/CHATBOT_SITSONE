# ui/streamlit_app.py
import json
import os
import sys
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import RAGConfig
from app.service.chatbot import build_app  # ajusta si tu archivo se llama distinto

load_dotenv("app/.env")

st.set_page_config(page_title="AMB Wiki", page_icon="💬", layout="centered")
st.title("AMB PRE")

cfg = RAGConfig.from_env()
app = build_app(cfg)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {}

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
prompt = st.chat_input("Escriu la teva pregunta… / Escribe tu pregunta…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscant resposta…"):
            res = app.handle_messages(
                st.session_state.messages,
                state=st.session_state.chat_state,
            )
            answer = res["answer"]
            st.markdown(answer)

            # Debug plegable
            with st.expander("Debug (scores / decisión)"):
                st.code(json.dumps(res, ensure_ascii=False, indent=2), language="json")

    st.session_state.chat_state = res.get("state") or {}
    st.session_state.messages.append({"role": "assistant", "content": answer})
