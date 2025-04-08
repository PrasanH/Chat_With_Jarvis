import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# OpenAIEmbeddings,
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  ### to chat with our text
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain_community.llms import HuggingFaceHub

# from InstructorEmbedding import INSTRUCTOR
from docx import Document
import os

import base64
from PIL import Image

from datetime import datetime
import json






def get_text_from_documents(uploaded_docs):
    """
    Functions returns text chunks for the uploaded files (.pdf or .docx)

    Args:
        uploaded_docs (_type_): _description_

    Returns:
        _type_: text chunks (RAW text chunks)
    """
    st.write(uploaded_docs)
    text = ""
    for uploaded_doc in uploaded_docs:

        if ".pdf" in uploaded_doc.name:
            pdf_reader = PdfReader(uploaded_doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif ".docx" in uploaded_doc.name:
            docx_text = extract_text_from_docx(uploaded_doc)
            text += docx_text
    return text


def extract_text_from_docx(docx_file):
    text = ""
    doc = Document(docx_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def get_text_chunks(raw_text):
    """
    function returns list of text chunks from raw text
    Here, we can specify the chunk size, overlap and the length function
    Args:
        raw_text (_type_): _description_

    Returns:
        _type_: list of text chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # char size
        chunk_overlap=150,  # to keep a bit of history so that to make sense, 200
        length_function=len,  # len function of python
    )
    chunks = text_splitter.split_text(raw_text)

    return chunks



def get_vectorstore(text_chunks:list):
    """
    Function returns vectorstore from FAISS from list of text chunks

    For embedding, we use OpenAI embediing model( default)
    
    Then, we use FAISS to create a vector store using the embedding model. ie. vector store where our text chunks are represented as numbers.

    Args:
        text_chunks (list): _description_

    Returns:
        _type_: _description_
    """
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore



def get_convo_chain(vectorstore, model="gpt-3.5-turbo"):
    """
    Function returns conversation chain

    Args:
        vectorstore: 
        model: default model is "gpt-3.5-turbo", if nothing is passed

    Returns:
        _type_: 
    """

    llm = ChatOpenAI(model=model)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature": 0.5, "max_length": 512},)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return convo_chain


def handle_user_input(user_question, conversation):
    # response = st.session_state.conversation({"question": user_question})
    response = conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("**Question:**", message.content)
        else:
            st.write("**Response:**", message.content)


def encode_image(uploaded_image):
    """Encodes an uploaded image to base64

    Args:
        uploaded_image (UploadedFile): The uploaded image file from Streamlit.

    Returns:
        str: Base64 encoded string of the image
    """
    image_bytes = uploaded_image.read()
    return base64.b64encode(image_bytes).decode('utf-8')
    


def display_uploaded_image(uploaded_image):
    """
    Displays the uploaded image from the user

    Args:
        uploaded_image (_type_): The uploaded image file from Streamlit.
    """
    image_to_display = Image.open(uploaded_image)
    return image_to_display


def save_chat_log(chat_history:list[str], name:str = None):
    """saves the chat log to a 'chat_log' folder 

    file_name is name_timestamp.txt if name is given else chat_timestamp.txt
    Args:
        chat_history (list[str]): list of chat history
        name (str, optional): optional file name. Defaults to None.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat_log_dir = os.path.join(base_dir, 'chat_log')
    os.makedirs(chat_log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"{name}_{timestamp}.txt" if name else f"chat_{timestamp}.txt"

    full_path = os.path.join(chat_log_dir, file_name)

    with open(full_path, 'w', encoding='utf-8') as f:
        prompt = chat_history[0]['content']
        question= chat_history[1]['content']
        response= chat_history[2]['content']

        f.write(f"prompt: {prompt}\n")
        f.write(f"question: {question}\n")
        f.write(f"response: {response}\n")
        #json.dump(chat_history, f, indent=4, ensure_ascii=False)
    print(f"Chat log saved to: {full_path}")