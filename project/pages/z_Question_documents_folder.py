import streamlit as st
import app_utils.folder_rag_utils as folder_rag_utils
from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDF Folder", page_icon="📁")

    st.header("📁 Ask Questions from PDFs in a Folder")
    st.write("This page allows you to query all PDFs in a specified folder using a persistent ChromaDB vector store.")


# Collection selection/creation
    st.subheader("Collection Management")
    
    # Get list of existing collections
    existing_collections = folder_rag_utils.list_collections()
    
    collection_option = st.radio(
        "Choose option:",
        ["Load Existing Collection", "Create New Collection"],
        horizontal=True
    )
    
    if collection_option == "Load Existing Collection":
        if existing_collections:
            collection_name = st.selectbox(
                "📂 Select Collection",
                options=existing_collections,
                help="Choose from previously created collections"
            )
        else:
            st.warning("⚠️ No existing collections found. Please create a new one.")
            collection_name = st.text_input(
                "🗃️ ChromaDB Collection Name",
                value="pdf_collection",
                help="Name for the vector database collection"
            )
    else:
        collection_name = st.text_input(
            "🗃️ ChromaDB Collection Name",
            value="pdf_collection",
            help="Name for the vector database collection"
        )
    
        # Folder selection
        pdf_folder = st.text_input(
            "📂 Enter PDF Folder Path",
            value="./pdf_documents",
            help="Path to folder containing PDF files"
        )

    # Model selection
    model = st.selectbox(
            label=":blue[Select model]",
            options=["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-5-2025-08-07", "gpt-5-mini-2025-08-07"],
            index=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Index/Update PDFs"):
            if os.path.exists(pdf_folder):
                with st.spinner("Processing PDFs and creating vector database..."):
                    num_docs = folder_rag_utils.index_pdf_folder(pdf_folder, collection_name)
                    st.success(f"✅ Indexed {num_docs} PDF documents successfully!")
            else:
                st.error(f"❌ Folder not found: {pdf_folder}")

    with col2:
        if st.button("📊 View Collection Stats"):
            stats = folder_rag_utils.get_collection_stats(collection_name)
            if stats:
                st.info(f"📚 Documents: {stats['count']}\n\n📄 Files: {stats['unique_files']}")

    with col3:
        if st.button("🗑️ Clear Collection"):
            folder_rag_utils.clear_collection(collection_name)
            st.warning("Collection cleared!")

    st.divider()

    # Question input
    user_question = st.text_area(
        "💬 Ask your question:",
        height=100,
        placeholder="Type your question about the documents..."
    )

    if st.button("🔍 Get Answer", type="primary"):
        if user_question:
            with st.spinner("Searching documents..."):
                response = folder_rag_utils.query_pdf_collection(
                    user_question, 
                    collection_name, 
                    model
                )
                
                if response:
                    st.subheader("📝 Answer:")
                    st.write(response['answer'])
                    
                    st.divider()
                    st.subheader("📚 Sources:")
                    
                    for i, source in enumerate(response['sources'], 1):
                        with st.expander(f"Source {i}: {source['file']} (Page {source['page']})"):
                            st.write(source['content'])
                            st.caption(f"Relevance Score: {source['score']:.3f}")
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()

    
