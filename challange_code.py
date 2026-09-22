# ep37_rag_pipeline.py
# CodeToAGI — Deep Learning EP37: Full RAG Pipeline

# 1. Install
# pip install langchain faiss-cpu sentence-transformers pypdf transformers torch

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ── Step 1: Load & Chunk ────────────────────────────────────────────────────
loader = PyPDFLoader("your-document.pdf")          # ← change this
pages  = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# ── Step 2: Embed & Store in FAISS ──────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# ── Step 3: Build Retriever + LLM ───────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=pipe)

# ── Step 4: Create RetrievalQA Chain ────────────────────────────────────────
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True   # useful for debugging
)

# ── Step 5: Query ───────────────────────────────────────────────────────────
result = qa_chain.invoke({"query": "What does this document say about X?"})
print(result["result"])

# Bonus: see the retrieved chunks
for i, doc in enumerate(result["source_documents"]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:300])
