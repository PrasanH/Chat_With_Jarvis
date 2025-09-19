import os
import openai
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import app_utils.llm_utils as llm_utils

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


pre_defined_content = [
    "You are an Intelligent assistant who is good at explaining things in a simple way",
    "Explain in simple words as if explaining it to a child",
    "You are an expert in programming",
    "Answer in minimum words as possible",
    "Answer in minimum words as possible with reasoning",
    "Check the grammar and rephrase if required. You are also allowed to improvise",
    "You are a helpful assistant",
    "I will give word(s). Just return suitable emojis and nothing else.",
    "Explain in simple words in Hinglish. Maintain a friendly tone. keep the text in english",
    "Explain in simple words in Kannada-English. Maintain a friendly tone. keep the text in english",
]

my_content = st.selectbox(
    label=" :red[Content]",
    options=pre_defined_content,
)
with st.expander(label="Type your content if needed :point_down:", expanded=False):
    typed_content = st.text_input("type your content")

if typed_content:
    my_content = typed_content


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
    


