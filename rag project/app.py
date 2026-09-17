import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load env
load_dotenv()

# -------------------- CACHE RAG --------------------
@st.cache_resource
def load_rag_chain():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Use ONLY the context below to answer.
    If not found, say "Analyzing sources , checking if information available....".

    Context:
    {context}

    Question:
    {input}
    """)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    doc_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, doc_chain)

rag_chain = load_rag_chain()

# -------------------- SESSION MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- UI --------------------
st.title("📚 Encyclopedia Q&A using RAG")
st.caption(
    "This project is for educational purposes only and does not provide medical advice."
)



# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # RAG response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]

    # Show assistant message
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # -------------------- SHOW SOURCES --------------------
    for i, doc in enumerate(response["context"], 1):
        with st.expander(f"Source {i}"):
            st.write("**Source:**", doc.metadata.get("source", "Unknown"))
            st.write("**Page:**", doc.metadata.get("page", "N/A"))
            st.write(doc.page_content[:500])  # preview first 500 chars

    # Memory control (keep last 6 messages)
    st.session_state.messages = st.session_state.messages[-6:]
