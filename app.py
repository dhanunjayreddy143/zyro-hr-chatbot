import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🏢",
    layout="centered",
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;
        color: white;
    }
    .source-badge {
        background: #e8f4f8; border-left: 3px solid #0f3460;
        padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.85rem;
        margin-top: 0.5rem; color: #333;
    }
    .oos-warning {
        background: #fff3cd; border-left: 3px solid #ffc107;
        padding: 0.5rem 1rem; border-radius: 4px;
    }
    .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics</h1>
    <h3>HR Help Desk Assistant</h3>
    <p>Ask me anything about HR policies, leave, benefits, and more!</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com"
    )
    corpus_path = st.text_input(
        "HR Corpus Path",
        value=os.environ.get("CORPUS_PATH", "/kaggle/input/zyro-dynamics-hr-corpus/"),
        help="Path to the folder containing the 11 HR policy PDFs"
    )
    st.markdown("---")
    st.markdown("**📂 Policy Documents Covered**")
    policies = [
        "🏛️ Company Profile",
        "📋 Employee Handbook",
        "🌴 Leave Policy",
        "🏠 Work From Home Policy",
        "⚖️ Code of Conduct",
        "📊 Performance Review",
        "💰 Compensation & Benefits",
        "🔒 IT & Data Security",
        "🛡️ POSH Policy",
        "🚪 Onboarding & Separation",
        "✈️ Travel & Expense",
    ]
    for p in policies:
        st.markdown(f"  {p}")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─── Prompts ───────────────────────────────────────────────────────────────────
OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Does the following question relate to HR policies,
leave, compensation, work-from-home, performance reviews, code of conduct,
IT security, POSH, onboarding, separation, or travel & expense at a company?

Answer with ONLY one word: YES or NO.

Question: {question}
Answer:
""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question accurately and concisely using ONLY the information
provided in the context below. Do NOT use any external knowledge.

If the context does not contain enough information to answer the question,
say: "I could not find specific information about this in the Zyro Dynamics policy documents."

Always mention which policy document your answer is based on when possible.

Context:
{context}

Question: {question}

Answer:
""")

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. "
    "Please reach out to the HR team directly for other inquiries."
)

# ─── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading HR documents and building knowledge base...")
def build_pipeline(corpus_path: str, groq_key: str):
    os.environ["GROQ_API_KEY"] = groq_key

    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)

    return retriever, llm, len(documents), len(chunks)


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown").split("/")[-1]
        formatted.append(f"[Source {i} - {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def ask_bot(question: str, retriever, llm) -> dict:
    """Guardrail → RAG pipeline."""
    # Step 1: classify
    classifier = OOS_PROMPT | llm | StrOutputParser()
    classification = classifier.invoke({"question": question}).strip().upper()

    if classification.startswith("NO"):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "out_of_scope": True}

    # Step 2: retrieve + generate
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = list(set(
        doc.metadata.get("source", "Unknown").split("/")[-1]
        for doc in docs
    ))
    return {"answer": answer, "sources": sources, "out_of_scope": False}


# ─── Main UI ───────────────────────────────────────────────────────────────────
if not groq_key:
    st.warning("👈 Please enter your Groq API key in the sidebar to get started.")
    st.stop()

if not os.path.isdir(corpus_path):
    st.error(f"❌ Corpus path not found: `{corpus_path}`. Please update the path in the sidebar.")
    st.stop()

retriever, llm, n_docs, n_chunks = build_pipeline(corpus_path, groq_key)

col1, col2 = st.columns(2)
col1.metric("📄 Documents Loaded", n_docs)
col2.metric("🧩 Chunks Created", n_chunks)
st.success("✅ HR knowledge base ready!")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I'm your Zyro Dynamics HR Assistant. 👋\n\n"
            "I can answer questions about leave policies, compensation, "
            "work-from-home rules, code of conduct, performance reviews, "
            "and much more — all grounded in official policy documents.\n\n"
            "How can I help you today?"
        ),
        "sources": [],
        "out_of_scope": False
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(
                f'<div class="source-badge">📚 <strong>Sources:</strong> {", ".join(msg["sources"])}</div>',
                unsafe_allow_html=True
            )
        if msg.get("out_of_scope"):
            st.markdown(
                '<div class="oos-warning">⚠️ This question is outside my HR policy scope.</div>',
                unsafe_allow_html=True
            )

# Chat input
if prompt := st.chat_input("Ask an HR question (e.g. 'How many sick leaves do I get?')"):
    st.session_state.messages.append({
        "role": "user", "content": prompt, "sources": [], "out_of_scope": False
    })
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy documents..."):
            result = ask_bot(prompt, retriever, llm)

        st.write(result["answer"])
        if result.get("sources"):
            st.markdown(
                f'<div class="source-badge">📚 <strong>Sources:</strong> {", ".join(result["sources"])}</div>',
                unsafe_allow_html=True
            )
        if result.get("out_of_scope"):
            st.markdown(
                '<div class="oos-warning">⚠️ This question is outside my HR policy scope.</div>',
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "out_of_scope": result.get("out_of_scope", False)
    })

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔒 Powered by RAG (FAISS + LLaMA 3.3 70B) | Answers grounded in Zyro Dynamics policy documents only")
