import streamlit as st
import app_utils.llm_utils as llm_utils
from dotenv import load_dotenv


def main():

    load_dotenv()  #### to read the .env file where you have stored the api keys

    st.set_page_config(page_title="Chat with PDFs/docs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header(":books: Upload your documents, click on process and start questioning")

    uploaded_docs = st.file_uploader(
        "Upload your **pdfs/docx**,type your question and click on process",
        accept_multiple_files=True,
        type=["pdf", "docx"],
    )
    

    model = st.selectbox(
        label=":blue[Select model]",
        options=["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-5-2025-08-07","gpt-5-mini-2025-08-07"],
        index=1,
        
    )

    if st.button("Process"):

        # Get raw text from uploaded docs ( .pdf or .docx) 

        raw_text = llm_utils.get_text_from_documents(uploaded_docs)
        # st.write(raw_text)   # debugging

        # Split the extracted raw text into smaller text chunks

        text_chunks = llm_utils.get_text_chunks(raw_text)
        #st.write(text_chunks)    # debugging

        # Create a vector store (embedding vector database) from the text chunks
        vectorstore = llm_utils.get_vectorstore(text_chunks)
        

        # Initialize the conversational retrieval chain with the vector store and model

        st.session_state.conversation = llm_utils.get_convo_chain(vectorstore, model)


    user_question = st.text_area(":blue[Type your question]")

    if user_question:
        llm_utils.handle_user_input(user_question, st.session_state.conversation)


if __name__ == "__main__":
    main()
