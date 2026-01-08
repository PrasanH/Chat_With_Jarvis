import os
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI


# Initialize ChromaDB client
def get_chroma_client():
    """Initialize persistent ChromaDB client"""
    return chromadb.PersistentClient(path="./chroma_db")


def extract_pdf_with_metadata(pdf_path: str) -> List[Dict]:
    """Extract text from PDF with page numbers and source file metadata"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    documents = []
    for page in pages:
        documents.append({
            'content': page.page_content,
            'metadata': {
                'source': os.path.basename(pdf_path),
                'page': page.metadata.get('page', 0) + 1,  # 1-indexed pages
                'full_path': pdf_path
            }
        })
    
    return documents


def index_pdf_folder(folder_path: str, collection_name: str = "pdf_collection") -> int:
    """
    Index all PDFs in a folder into ChromaDB
    Returns: number of documents indexed
    """
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    
    if not pdf_files:
        return 0
    
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Extract and process all PDFs
    for pdf_file in pdf_files:
        docs = extract_pdf_with_metadata(str(pdf_file))
        
        for doc in docs:
            # Split text into chunks while preserving metadata
            chunks = text_splitter.split_text(doc['content'])
            
            for chunk in chunks:
                all_documents.append({
                    'content': chunk,
                    'metadata': doc['metadata']
                })
    
    # Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    
    # Prepare data for ChromaDB
    texts = [doc['content'] for doc in all_documents]
    metadatas = [doc['metadata'] for doc in all_documents]
    
    # Create or update ChromaDB collection
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    return len(all_documents)


def query_pdf_collection(
    question: str, 
    collection_name: str = "pdf_collection",
    model: str = "gpt-4.1-mini-2025-04-14",
    k: int = 4
) -> Optional[Dict]:
    """
    Query the PDF collection and return answer with sources
    """
    try:
        embeddings = OpenAIEmbeddings()
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Custom prompt to include source citations
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        When providing your answer, reference the specific sources (document name and page number) where you found the information.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer with source citations:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        llm = ChatOpenAI(model_name=model, temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = qa_chain({"query": question})
        
        # Format sources
        sources = []
        for doc in result['source_documents']:
            sources.append({
                'file': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'content': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                'score': 1.0  # ChromaDB similarity score (you can add this with similarity_search_with_score)
            })
        
        return {
            'answer': result['result'],
            'sources': sources
        }
        
    except Exception as e:
        print(f"Error querying collection: {e}")
        return None


def get_collection_stats(collection_name: str = "pdf_collection") -> Optional[Dict]:
    """Get statistics about the collection"""
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=collection_name)
        
        data = collection.get()
        
        unique_files = set()
        if data['metadatas']:
            for metadata in data['metadatas']:
                unique_files.add(metadata.get('source', 'Unknown'))
        
        return {
            'count': collection.count(),
            'unique_files': len(unique_files)
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return None


def clear_collection(collection_name: str = "pdf_collection"):
    """Clear/delete a collection"""
    try:
        client = get_chroma_client()
        client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Error clearing collection: {e}")


def list_collections():
    """List all existing ChromaDB collections"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        return []