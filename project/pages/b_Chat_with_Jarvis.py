import os
import openai
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import app_utils.llm_utils as llm_utils
from app_utils import content

load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = my_api_key)


st.set_page_config(page_title="Chat with JARVIS", page_icon=":robot_face:")

st.header(":robot_face: Chat with JARVIS")

if "chat_history_jarvis" not in st.session_state:
        st.session_state.chat_history_jarvis = []

@st.fragment
def save_chat():
    submitted = st.checkbox('save_chat')
    if submitted:
        llm_utils.save_chat_log(st.session_state.chat_history_jarvis[-3:])
        st.write('chat saved successfully')


my_content = st.selectbox(
    label=" :red[Content]",
    options=content.pre_defined_content,
)

with st.expander(label="Type System prompt if needed :point_down:", expanded=False):
    system_prompt = st.text_input("type your system prompt here")

if system_prompt:
    my_content = system_prompt


st.session_state.chat_history_jarvis.append(
    {
        "role": "system",
        "content": my_content,
    },
)
model = st.selectbox(
    label=":blue[Select model]",
    options=["gpt-5-2025-08-07","gpt-5-mini-2025-08-07", "gpt-4.1-mini-2025-04-14","gpt-4o", "gpt-4o-mini", "gpt-4.1-2025-04-14", "chatgpt-4o-latest", "gpt-4o-mini-2024-07-18", "gpt-4o-mini-search-preview", "gpt-4o-search-preview"],
    index = 2,
)


question = st.text_area(":red[Type your question]", height=300)
message = f"User : {question}"

if question:
    st.session_state.chat_history_jarvis.append(
        {"role": "user", "content": message},
    )
    chat = client.chat.completions.create(model=model, messages=st.session_state.chat_history_jarvis)

    reply = chat.choices[0].message.content

    if reply:
        st.write(f":robot_face:  {reply}")
    print(f"JARVIS: {reply}")
    st.session_state.chat_history_jarvis.append({"role": "assistant", "content": reply})
    
    save_chat()
    


