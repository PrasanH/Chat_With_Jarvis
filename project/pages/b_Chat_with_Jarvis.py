import os
import openai
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = my_api_key)


st.set_page_config(page_title="Chat with JARVIS", page_icon=":robot_face:")

st.header(":robot_face: Chat with JARVIS")

if "chat_history_jarvis" not in st.session_state:
        st.session_state.chat_history_jarvis = []


pre_defined_content = [
    "You are an Intelligent assistant who is good at explaining things in a simple way",
    "You are an expert in programming",
    "You are the most intelligent assistant who can solve anything",
    "Answer in minimum words as possible",
    "Answer in minimum words as possible with reasoning",
    "You are a Scientific research student. Rephrase this sentence to avoid plagiarism",
    "Check the grammar and rephrase if required. You are also allowed to improvise",
    "You are a helpful assistant"
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
    options=["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-search-preview", "gpt-4o-search-preview"],
    index = 1,
)


question = st.text_area(":red[Type your question]")
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
    



