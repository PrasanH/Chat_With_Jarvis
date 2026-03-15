import streamlit as st
import chromadb
import json

st.header("ChromaDB Contents Viewer")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get all collections
collections = client.list_collections()

st.write(f"**Total Collections:** {len(collections)}")

for collection in collections:
    # list_collections() returns full Collection objects — no need to call get_collection() again
    # Fetch all metadata entries for this collection (no documents/embeddings needed)
    results = collection.get(include=["metadatas"])

    # Derive chunk count from fetched results — avoids a separate coll.count() DB call
    chunk_count = len(results["metadatas"])

    # Different loaders store the source file name under different keys.
    # Try "source" first (LangChain default), then fall back to other common key names.
    # Build a deduplicated set, remove empty strings, then sort alphabetically.
    pdf_names = sorted(
        {
            m.get("source")  # LangChain default key
            or m.get("file_name")  # some custom loaders
            or m.get("filename")  # others
            or m.get("pdf_name")  # or this
            or ""  # fallback so the value is never None
            for m in results["metadatas"]
            if m
        }
        - {""}  # drop the empty-string fallback from the set
    )

    pdf_count = len(pdf_names)

    # Show chunk count and PDF count in the expander header for a quick overview
    expander_label = (
        f"Collection: {collection.name}  —  {chunk_count} chunks  |  {pdf_count} PDF(s)"
    )

    with st.expander(expander_label):
        st.write(f"**Total Chunks:** {chunk_count}")
        st.write(f"**Total PDFs:** {pdf_count}")

        # List every unique PDF file found in this collection's metadata
        if pdf_names:
            st.write("**PDF Files:**")
            for name in pdf_names:
                st.write(f"- {name}")
        else:
            st.info("No PDF source info found in metadata.")

        # Metadata is only fetched/displayed on demand to keep the UI clean
        if st.button("Show Metadata", key=f"meta_{collection.name}"):
            if results["metadatas"]:
                # Use json string as a hashable key for O(n) deduplication instead of O(n²) list scan
                seen = set()
                for metadata in results["metadatas"]:
                    key = json.dumps(metadata, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        st.json(metadata)
            else:
                st.info("No metadata found in this collection.")
