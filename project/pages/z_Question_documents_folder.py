import streamlit as st
import app_utils.folder_rag_utils as folder_rag_utils
from dotenv import load_dotenv
import os


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDF Folder", page_icon="📁")

    st.header("📁 Ask Questions from PDFs in a Folder")
    st.write(
        "This page allows you to query all PDFs in a specified folder using a persistent ChromaDB vector store."
    )

    # Collection selection/creation
    st.subheader("Collection Management")

    # Get list of existing collections
    existing_collections = folder_rag_utils.list_collections()

    collection_option = st.radio(
        "Choose option:",
        ["Load Existing Collection", "Create New Collection", "Delete Collection"],
        horizontal=True,
    )

    collection_name = None

    if collection_option == "Load Existing Collection":
        if existing_collections:
            collection_name = st.selectbox(
                "📂 Select Collection",
                options=existing_collections,
                help="Choose from previously created collections",
            )
        else:
            st.warning("⚠️ No existing collections found. Please create a new one.")

    elif collection_option == "Create New Collection":
        collection_name = st.text_input(
            "🗃️ ChromaDB Collection Name",
            value="pdf_collection",
            help="Name for the vector database collection",
        )
        pdf_folder = st.text_input(
            "📂 Enter PDF Folder Path",
            value="./pdf_documents",
            help="Path to folder containing PDF files",
        )

    else:  # Delete Collection
        if existing_collections:
            collection_to_delete = st.selectbox(
                "🗑️ Select Collection to Delete",
                options=existing_collections,
                help="Choose the collection you want to delete",
            )
            if st.button("🗑️ Delete Collection", type="primary"):
                folder_rag_utils.clear_collection(collection_to_delete)
                st.warning(f"Collection '{collection_to_delete}' deleted!")
        else:
            st.warning("⚠️ No existing collections found.")

    # Model + actions only relevant for Load and Create
    if collection_option != "Delete Collection":
        model = st.selectbox(
            label=":blue[Select model]",
            options=[
                "gpt-4.1-2025-04-14",
                "gpt-4.1-mini-2025-04-14",
                "gpt-5-2025-08-07",
                "gpt-5-mini-2025-08-07",
            ],
            index=1,
        )

        if collection_option == "Create New Collection":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Index/Update PDFs"):
                    if os.path.exists(pdf_folder):
                        progress_placeholder = st.empty()

                        def on_progress(pdf_name: str):
                            progress_placeholder.info(f"📄 Processing: **{pdf_name}**")

                        with st.spinner(
                            "Processing PDFs and creating vector database..."
                        ):
                            result = folder_rag_utils.index_pdf_folder(
                                pdf_folder,
                                collection_name,
                                progress_callback=on_progress,
                            )

                        progress_placeholder.empty()
                        st.success(
                            f"✅ Indexed **{result['num_files']}** PDF files | "
                            f"**{result['num_pages']}** pages | "
                            f"**{result['num_chunks']}** chunks"
                        )
                    else:
                        st.error(f"❌ Folder not found: {pdf_folder}")

            with col2:
                if st.button("📊 View Collection Stats"):
                    stats = folder_rag_utils.get_collection_stats(collection_name)
                    if stats:
                        st.info(
                            f"📚 Chunks: {stats['count']}\n\n📄 Files: {stats['unique_files']}"
                        )

        elif collection_option == "Load Existing Collection" and collection_name:
            if st.button("📊 View Collection Stats"):
                stats = folder_rag_utils.get_collection_stats(collection_name)
                if stats:
                    st.info(
                        f"📚 Chunks: {stats['count']}\n\n📄 Files: {stats['unique_files']}"
                    )

        st.divider()

        user_question = st.text_area(
            "💬 Ask your question:",
            height=100,
            placeholder="Type your question about the documents...",
        )

        if st.button("🔍 Get Answer", type="primary"):
            if user_question:
                with st.spinner("Searching documents..."):
                    response = folder_rag_utils.query_pdf_collection(
                        user_question, collection_name, model
                    )

                    if response:
                        st.subheader("📝 Answer:")
                        st.write(response["answer"])

                        st.divider()
                        with st.expander("📊 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(
                                    f"**Source {i}:** `{source['file']}` (Page {source['page']})"
                                )
                                st.write(source["content"])
                                st.caption(f"Relevance Score: {source['score']:.3f}")
                                st.divider()
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
