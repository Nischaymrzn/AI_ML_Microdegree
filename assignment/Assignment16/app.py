import streamlit as st
import rag_chat as rag

st.set_page_config(page_title="Web RAG Chatbot 🤖")

st.title("🤖 Transformer RAG Chatbot")
st.caption("Ask questions from multiple AI/ML web sources")

# Load pipeline
@st.cache_resource
def load():
    chunks, sources = rag.load_data()
    embedder, index = rag.build_index(chunks)
    generator = rag.load_model()

    return chunks, sources, embedder, index, generator

chunks, sources, embedder, index, generator = load()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for m in st.session_state.messages:
    st.write(f"**{m['role']}**: {m['text']}")

# Input
q = st.chat_input("Ask something about Transformers...")

if q:
    st.session_state.messages.append({"role": "You", "text": q})

    hits = rag.retrieve(q, embedder, index, chunks, sources)

    st.subheader("🔍 Evidence")
    for i, h in enumerate(hits):
        st.write(f"Source {i+1}: {h[1]}")
        st.write(h[0][:200] + "...")

    ans = rag.answer(q, hits, generator)

    st.subheader("🤖 Answer")
    st.write(ans)

    st.session_state.messages.append({"role": "AI", "text": ans})