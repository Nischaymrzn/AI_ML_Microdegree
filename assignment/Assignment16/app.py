import streamlit as st
import importlib.util
from pathlib import Path

def _load_rag_module():
    """Dynamically load rag_web_engine.py so we can use its functions."""
    root = Path(__file__).resolve().parent
    path = root / "rag_chat.py"
    spec = importlib.util.spec_from_file_location("rag_web_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@st.cache_resource(show_spinner="Loading pages, embeddings & index...")
def load_pipeline():
    """
    Build the RAG pipeline and cache it.
    @st.cache_resource makes sure we don't re-download pages
    and re-build the index every time the user sends a message.
    """
    rag = _load_rag_module()
    pipe = rag.load_pipeline()
    pipe["rag"] = rag  # keep reference to module functions
    return pipe

# Main app

def main():
    # --- Page setup ---
    st.set_page_config(
        page_title="Transformer RAG Chatbot",
        page_icon="",
        layout="centered",
    )

    st.title("Transformer RAG Chatbot")
    st.caption(
        "Ask questions about **Transformers** & **Attention Mechanisms**. "
        "Answers are retrieved from different websites using RAG."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        k = st.slider("Number of chunks to retrieve (k)", 1, 10, 3)
        show_evidence = st.checkbox("Show retrieved evidence", value=True)

        st.divider()
        st.markdown("### Knowledge Sources")
        st.markdown(
            "1. [GeeksForGeeks - Transformers](https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/)  \n"
            "2. [DataCamp - How Transformers Work](https://www.datacamp.com/tutorial/how-transformers-work)  \n"
            "3. [AWS - What are Transformers](https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/)  \n"
            "4. [IBM - Transformer Model](https://www.ibm.com/think/topics/transformer-model)  \n"
            "5. [Medium - Transformers](https://medium.com/data-science/transformers-89034557de14)  \n"
            "6. [Serokell - Transformers in ML](https://serokell.io/blog/transformers-in-ml)"
        )

        st.divider()
        st.markdown(
            "**How it works:**\n"
            "1. Web pages are fetched & cleaned\n"
            "2. Text is split into overlapping chunks\n"
            "3. Chunks are embedded into vectors\n"
            "4. FAISS finds the most similar chunks\n"
            "5. Answer is generated from evidence"
        )

    # --- Load pipeline ---
    pipe = load_pipeline()
    rag = pipe["rag"]

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    prompt = st.chat_input("Ask about Transformers & Attention...")

    if not prompt:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Retrieve & answer ---
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            hits = rag.retrieve(
                prompt, pipe["embedder"], pipe["index"], pipe["chunks"], k=k
            )

        # Show evidence if enabled
        if show_evidence and hits:
            with st.expander(f"📎 Retrieved Evidence ({len(hits)} chunks)", expanded=False):
                for i, h in enumerate(hits, 1):
                    from urllib.parse import urlparse
                    domain = urlparse(h["source"]).netloc.replace("www.", "")
                    st.markdown(
                        f"**Source {i}** — _{domain}_ "
                        f"(similarity: `{h['score']:.2f}`)"
                    )
                    st.info(h["text"][:300] + "...")
                    st.markdown("---")

        # Generate and stream the answer word-by-word (like ChatGPT)
        answer = rag.generate_answer(prompt, hits)

        def stream_answer():
            """Generator that yields words one at a time for streaming effect."""
            import time
            for word in answer.split(" "):
                yield word + " "
                time.sleep(0.03)

        st.write_stream(stream_answer())

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()