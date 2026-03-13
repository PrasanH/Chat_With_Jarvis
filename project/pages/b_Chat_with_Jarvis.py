import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from datetime import datetime
import app_utils.llm_utils as llm_utils
from app_utils import content

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="Chat with JARVIS", page_icon=":robot_face:", layout="wide"
)

# --- Session state initialisation ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None  # None = unsaved new chat
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "session_title" not in st.session_state:
    st.session_state.session_title = "New Chat"
if "session_created" not in st.session_state:
    st.session_state.session_created = None


# --- Helpers ---
def start_new_chat():
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
    st.session_state.session_created = None


def load_session(session_id: str):
    data = llm_utils.load_chat_session(session_id)
    if data:
        st.session_state.session_id = data["id"]
        st.session_state.messages = data["messages"]
        st.session_state.session_title = data.get("title", "Untitled")
        st.session_state.session_created = data.get("created")


def delete_session(session_id: str):
    llm_utils.delete_chat_session(session_id)
    if st.session_state.session_id == session_id:
        start_new_chat()


# --- Sidebar: chat history ---
with st.sidebar:
    st.title(":speech_balloon: Chat History")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()

    sessions = llm_utils.list_chat_sessions()
    if sessions:
        for session in sessions:
            is_active = st.session_state.session_id == session["id"]
            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"**{session['title']}**" if is_active else session["title"]
                if st.button(
                    label, key=f"sel_{session['id']}", use_container_width=True
                ):
                    load_session(session["id"])
                    st.rerun()
            with col2:
                if st.button(":wastebasket:", key=f"del_{session['id']}"):
                    delete_session(session["id"])
                    st.rerun()
    else:
        st.caption("No saved chats yet.")


# --- Main area ---
st.header(":robot_face: Chat with JARVIS")

with st.expander(":gear: Settings", expanded=False):
    model = st.selectbox(
        "Model",
        options=[
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1-2025-04-14",
            "chatgpt-4o-latest",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini-search-preview",
            "gpt-4o-search-preview",
        ],
        index=2,
        key="model_select",
    )
    preset = st.selectbox(
        "System prompt preset",
        options=content.pre_defined_content,
        key="preset_select",
    )
    custom_prompt = st.text_area(
        "Or type a custom system prompt (overrides preset)", key="custom_prompt"
    )

# Determine active system prompt from session state
_custom = st.session_state.get("custom_prompt", "").strip()
system_prompt = (
    _custom
    if _custom
    else st.session_state.get("preset_select", content.pre_defined_content[0])
)

# --- Display current conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if user_input := st.chat_input("Ask JARVIS anything..."):
    # Show and store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build API payload (system prompt + conversation)
    api_messages = [
        {"role": "system", "content": system_prompt}
    ] + st.session_state.messages

    # Get and display reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=st.session_state.get("model_select", "gpt-4.1-mini-2025-04-14"),
                messages=api_messages,
            )
            reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Auto-save: create a new session ID on the first message
    if st.session_state.session_id is None:
        st.session_state.session_id = (
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        st.session_state.session_created = datetime.now().isoformat()
        st.session_state.session_title = user_input[:50] + (
            "..." if len(user_input) > 50 else ""
        )

    llm_utils.save_chat_session(
        session_id=st.session_state.session_id,
        title=st.session_state.session_title,
        messages=st.session_state.messages,
        created=st.session_state.session_created,
    )

    st.rerun()
