import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)

# Vector DB
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    "vectorstore/db_faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I only answer questions related to healthcare".

Context:
{context}

Question:
{input}
""")

# Chains
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever(search_kwargs={"k": 3})

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)


query = input("Write Query Here: ")
response = rag_chain.invoke({"input": query})


print("\nANSWER:\n", response["answer"])


print("\nSOURCE DOCUMENTS:\n")
for i, doc in enumerate(response["context"], 1):
    print(f"--- Source {i} ---")
    print("Metadata:", doc.metadata)
    print("Content preview:", doc.page_content[:300])
    print()
