import streamlit as st
import chromadb
from chromadb.config import Settings

st.header("ChromaDB Contents Viewer")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Get all collections
collections = client.list_collections()

st.write(f"**Total Collections:** {len(collections)}")

for collection in collections:
    with st.expander(f"Collection: {collection.name}"):
        # Get collection details
        coll = client.get_collection(collection.name)
        
        # Get all items
        results = coll.get()
        
        st.write(f"**Total Documents:** {coll.count()}")
        
        # Display metadata
        if results['metadatas']:
            st.write("**Metadata:**")
            for i, metadata in enumerate(results['metadatas']):
                st.json(metadata)
                
        # Display document IDs
        if results['ids']:
            st.write("**Document IDs:**")
            st.write(results['ids'])