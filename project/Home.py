import streamlit as st

st.set_page_config(page_title="Hello Jarvis", page_icon=":robot_face:")

st.title("Hello Sir, How Can I Help You Today? :sunglasses:")
st.write('### :point_left: Please select app page from the left')

st.write(':blue[1. Question documents:] Ask questions on your multiple or single PDF/docx files.')
st.write(':blue[2. Chat with Jarvis:] Ask LLMs(GPT/Gemini) direct questions including images and PDF files. Select predefined content or customized content. Select reasoning levels if required.')
st.write(':blue[3. Question Images:] Ask questions on your images')
st.write(':blue[4. Question Document folder:] Ask questions on all PDF files in a folder with persistent vector database')

st.sidebar.success("Select a page")
