import streamlit as st
import chromadb

st.header("ChromaDB Contents Viewer")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get all collections
collections = client.list_collections()

st.write(f"**Total Collections:** {len(collections)}")

for collection in collections:
    coll = client.get_collection(collection.name)
    chunk_count = coll.count()

    # Extract PDF names from metadata upfront
    results = coll.get(include=["metadatas"])
    pdf_names = sorted(
        {
            m.get("source")
            or m.get("file_name")
            or m.get("filename")
            or m.get("pdf_name")
            or ""
            for m in (results["metadatas"] or [])
            if m
        }
        - {""}
    )

    pdf_count = len(pdf_names)
    expander_label = (
        f"Collection: {collection.name}  —  {chunk_count} chunks  |  {pdf_count} PDF(s)"
    )

    with st.expander(expander_label):
        st.write(f"**Total Chunks:** {chunk_count}")
        st.write(f"**Total PDFs:** {pdf_count}")

        if pdf_names:
            st.write("**PDF Files:**")
            for name in pdf_names:
                st.write(f"- {name}")
        else:
            st.info("No PDF source info found in metadata.")

        if st.button("Show Metadata", key=f"meta_{collection.name}"):
            if results["metadatas"]:
                seen = []
                for metadata in results["metadatas"]:
                    if metadata not in seen:
                        seen.append(metadata)
                        st.json(metadata)
            else:
                st.info("No metadata found in this collection.")
