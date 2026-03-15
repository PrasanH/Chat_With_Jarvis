import os
from dotenv import load_dotenv
import streamlit as st
import app_utils.llm_utils as llm_utils
import app_utils.content as content
from app_utils.config import models
from google import genai

load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")
google_client = genai.Client(api_key=google_api_key)


st.set_page_config(page_title="JARVIS Gemini", page_icon="🤖")

st.header("🤖 Chat with JARVIS Gemini")

if "chat_history_jarvis" not in st.session_state:
    st.session_state.chat_history_jarvis = []


@st.fragment
def save_chat():
    submitted = st.checkbox("save_chat")
    if submitted:
        llm_utils.save_chat_log(st.session_state.chat_history_jarvis[-3:])
        st.write("chat saved successfully")


system_prompt_key = st.selectbox(
    label=" :red[System Prompt]",
    options=content.pre_defined_content.keys(),
)
system_prompt = content.pre_defined_content[system_prompt_key]

with st.expander(label="Type system prompt if needed :point_down:", expanded=False):
    typed_system_prompt = st.text_input("type system prompt here :")

if typed_system_prompt:
    system_prompt = typed_system_prompt


model = st.selectbox(
    label=":blue[Select model]",
    options=models["Gemini models"],
    index=0,
)


question = st.text_area(":red[Type your question]", height=300)
message = f"{question}"


send_button = st.button("Send", disabled=not question.strip())

if send_button and question:

    st.session_state.chat_history_jarvis.append(
        {"role": "user", "content": system_prompt + ". " + message},
    )
    # st.write(st.session_state.chat_history_jarvis)
    with st.spinner("🤖 Jarvis is working..."):
        chat = google_client.interactions.create(
            model=model, input=st.session_state.chat_history_jarvis
        )
        # st.write(chat.outputs)
        reply = chat.outputs[-1].text

    if reply:
        st.write(f"🤖  {reply}")
    print(f"JARVIS: {reply}")
    st.session_state.chat_history_jarvis.append({"role": "model", "content": reply})

    save_chat()


# st.write(st.session_state.chat_history_jarvis)
