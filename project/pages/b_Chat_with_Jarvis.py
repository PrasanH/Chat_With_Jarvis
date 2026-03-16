import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from datetime import datetime
from google import genai
import app_utils.llm_utils as llm_utils
import app_utils.folder_rag_utils as rag
from app_utils import content
from app_utils.config import models, gpt_default

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []  # filenames of indexed PDFs
if "session_has_docs" not in st.session_state:
    st.session_state.session_has_docs = False
if "renaming_session_id" not in st.session_state:
    st.session_state.renaming_session_id = None  # ID of session being renamed
if "pending_image" not in st.session_state:
    st.session_state.pending_image = (
        None  # base64 str of image to attach to next message
    )
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # increment to reset file uploaders


# --- Helpers ---
def start_new_chat():
    """Start a blank unsaved chat."""
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
    st.session_state.session_created = None
    st.session_state.session_docs = []
    st.session_state.session_has_docs = False
    st.session_state.pending_image = None
    st.session_state.uploader_key += 1


def load_session(session_id: str):
    """Load a saved session into state."""
    data = llm_utils.load_chat_session(session_id)
    if data:
        st.session_state.session_id = data["id"]
        st.session_state.messages = data["messages"]
        st.session_state.session_title = data.get("title", "Untitled")
        st.session_state.session_created = data.get("created")
        st.session_state.session_docs = data.get("docs", [])
        st.session_state.session_has_docs = len(st.session_state.session_docs) > 0
        st.session_state.pending_image = None
        st.session_state.uploader_key += 1


def delete_session(session_id: str):
    """Delete a session and its associated Chroma collection."""
    rag.delete_collection(session_id)  # remove associated Chroma collection
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
        with st.container(height=220, border=False):
            for session in sessions:
                is_active = st.session_state.session_id == session["id"]
                is_renaming = st.session_state.renaming_session_id == session["id"]

                if is_renaming:
                    new_title = st.text_input(
                        "Rename",
                        value=session["title"],
                        key=f"rename_input_{session['id']}",
                        label_visibility="collapsed",
                    )
                    rc1, rc2 = st.columns([1, 1])
                    with rc1:
                        if st.button(
                            "Save",
                            key=f"rename_save_{session['id']}",
                            use_container_width=True,
                        ):
                            llm_utils.rename_chat_session(
                                session["id"], new_title.strip() or session["title"]
                            )
                            if st.session_state.session_id == session["id"]:
                                st.session_state.session_title = (
                                    new_title.strip() or session["title"]
                                )
                            st.session_state.renaming_session_id = None
                            st.rerun()
                    with rc2:
                        if st.button(
                            "Cancel",
                            key=f"rename_cancel_{session['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.renaming_session_id = None
                            st.rerun()
                else:
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        label = (
                            f"**{session['title']}**" if is_active else session["title"]
                        )
                        if st.button(
                            label, key=f"sel_{session['id']}", use_container_width=True
                        ):
                            load_session(session["id"])
                            st.rerun()
                    with col2:
                        if st.button(":pencil2:", key=f"ren_{session['id']}"):
                            st.session_state.renaming_session_id = session["id"]
                            st.rerun()
                    with col3:
                        if st.button(":wastebasket:", key=f"del_{session['id']}"):
                            delete_session(session["id"])
                            st.rerun()
    else:
        st.caption("No saved chats yet.")


# --- Main area ---
st.header(":robot_face: Chat with JARVIS")

with st.expander(":gear: Settings", expanded=False):
    provider = st.radio(
        ":blue[Provider]",
        options=["GPT", "Gemini"],
        horizontal=True,
        key="provider_select",
    )
    _model_options = (
        models["GPT models"]
        if st.session_state.get("provider_select", "GPT") == "GPT"
        else models["Gemini models"]
    )
    _default_idx = (
        models["GPT models"].index(gpt_default)
        if st.session_state.get("provider_select", "GPT") == "GPT"
        else 0
    )
    model = st.selectbox(
        ":blue[Model]",
        options=_model_options,
        index=_default_idx,
        key="model_select",
    )
    preset = st.selectbox(
        ":blue[System prompt preset]",
        options=list(content.pre_defined_content.keys()),
        key="preset_select",
    )
    custom_prompt = st.text_area(
        ":blue[Or type a custom system prompt (overrides preset)]", key="custom_prompt"
    )
    if st.session_state.get(
        "provider_select", "GPT"
    ) == "GPT" and llm_utils._model_supports_reasoning(
        st.session_state.get("model_select", gpt_default)
    ):
        st.selectbox(
            "Reasoning effort",
            options=["low", "medium", "high"],
            index=0,
            key="reasoning_effort",
            help="Only available for GPT-5+ models.",
        )

# Determine active system prompt from session state
_custom = st.session_state.get("custom_prompt", "").strip()
_preset_key = st.session_state.get(
    "preset_select", next(iter(content.pre_defined_content))
)
system_prompt = _custom if _custom else content.pre_defined_content[_preset_key]

# --- Display current conversation ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    st.image(part["image_url"]["url"], width=300)
        else:
            st.markdown(msg["content"])

# --- Attachments ---
with st.expander(":paperclip: Attach files", expanded=False):
    att_col1, att_col2 = st.columns(2)

    with att_col1:
        st.caption(":page_facing_up: **Documents (PDF)**")
        uploaded_files = st.file_uploader(
            "Upload PDF(s) to chat context",
            type="pdf",
            accept_multiple_files=True,
            key=f"doc_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
        if uploaded_files:
            new_files = [
                f for f in uploaded_files if f.name not in st.session_state.session_docs
            ]
            if new_files:
                if st.session_state.session_id is None:
                    st.session_state.session_id = (
                        f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    )
                    st.session_state.session_created = datetime.now().isoformat()
                    st.session_state.session_title = "Doc Chat"
                with st.spinner("Indexing documents..."):
                    new_names = rag.index_uploaded_pdfs(
                        uploaded_files=new_files,
                        collection_name=st.session_state.session_id,
                    )
                for name in new_names:
                    if name not in st.session_state.session_docs:
                        st.session_state.session_docs.append(name)
                st.session_state.session_has_docs = True
                llm_utils.save_chat_session(
                    session_id=st.session_state.session_id,
                    title=st.session_state.session_title,
                    messages=st.session_state.messages,
                    created=st.session_state.session_created,
                    docs=st.session_state.session_docs,
                )
                st.success(f"Indexed {len(new_names)} file(s)")
        if st.session_state.session_docs:
            for doc_name in st.session_state.session_docs:
                st.caption(f":page_facing_up: {doc_name}")
        else:
            st.caption("No documents attached.")

    with att_col2:
        st.caption(":frame_with_picture: **Image**")
        uploaded_img = st.file_uploader(
            "Attach an image",
            type=["jpg", "jpeg", "png"],
            key=f"img_upload_{st.session_state.uploader_key}",
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded_img:
            st.session_state.pending_image = llm_utils.encode_image(uploaded_img)
            # encode_image returns a full data URI; pass it directly to st.image
            st.image(st.session_state.pending_image, width=200)
        elif not st.session_state.pending_image:
            st.caption("No image attached.")

# --- Chat input ---
if user_input := st.chat_input("Ask JARVIS anything..."):
    # Build user message content — multimodal if an image is attached
    if st.session_state.pending_image:
        user_content = [
            {"type": "text", "text": user_input},
            {
                "type": "image_url",
                "image_url": {
                    "url": st.session_state.pending_image  # full data URI with correct MIME type
                },
            },
        ]
        st.session_state.pending_image = None
    else:
        user_content = user_input

    # Show and store user message
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        if isinstance(user_content, list):
            st.markdown(user_content[0]["text"])
            st.image(user_content[1]["image_url"]["url"], width=250)
        else:
            st.markdown(user_content)

    # Build API payload — inject RAG context if docs are attached
    if st.session_state.session_has_docs:
        chunks = rag.retrieve_context(user_input, st.session_state.session_id)
        if chunks:
            ctx_text = "\n\n".join(
                f"[{c['source']} p.{c['page']}]: {c['content']}" for c in chunks
            )
            rag_system = (
                f"{system_prompt}\n\n"
                "Use the document excerpts below to answer when relevant. "
                "Cite the source file and page number.If you do not find the answer, say 'I don't know'.\n\n"
                f"Document context:\n{ctx_text}"
            )
        else:
            rag_system = system_prompt
    else:
        rag_system = system_prompt

    api_messages = [
        {"role": "system", "content": rag_system}
    ] + st.session_state.messages

    # Get and display reply
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            reply = llm_utils.get_llm_reply(
                client=client,
                model=st.session_state.get("model_select", gpt_default),
                messages=api_messages,
                reasoning_effort=st.session_state.get("reasoning_effort", "low"),
                gemini_client=gemini_client,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Auto-save: create a new session ID on the first message
    if st.session_state.session_id is None:
        st.session_state.session_id = (
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        st.session_state.session_created = datetime.now().isoformat()
        title_text = user_input if isinstance(user_content, str) else user_input
        st.session_state.session_title = title_text[:50] + (
            "..." if len(title_text) > 50 else ""
        )

    llm_utils.save_chat_session(
        session_id=st.session_state.session_id,
        title=st.session_state.session_title,
        messages=st.session_state.messages,
        created=st.session_state.session_created,
        docs=st.session_state.session_docs,
    )

    st.rerun()
