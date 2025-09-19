import os
import openai
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv()

my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = my_api_key)


st.set_page_config(page_title="JARVIS reasoning(o1-mini)", page_icon= ":robot_face:")

st.header(":robot_face: JARVIS Reasoning :bulb: ")

model = st.selectbox(
    label=":blue[Select model]",
    options=["o3-mini", "o1-mini"],
    index = 0,
)


my_content = st.text_area(":red[Type your question]")
text_to_append = "Answer in brief"
#o1 models are expensive. to reduce the token size we will prefer the brief answers

brief_answer = st.checkbox("Answer in Brief", value= True)


if brief_answer:
    my_content = my_content + '. ' + text_to_append
    #st.write('new_text:', my_content)

if my_content != '. ' + text_to_append:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": my_content,
            }
        ]
    )

    st.write(response.choices[0].message.content)
