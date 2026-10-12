"""
The RAG pipeline: ingestion -> chunking -> embeddings -> search -> answer.
"""
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()  # reads the .env file

# ---------- 1. Embeddings: turn text into numbers that capture meaning ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------- 2. Vector store: a database that stores those numbers ----------
if os.getenv("VECTOR_STORE", "chroma") == "pinecone":
    from langchain_pinecone import PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings
    )
else:
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db",  # saved on disk, survives restarts
    )

# ---------- 3. The LLM that writes the final answer ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 0 = stick to facts


# ===================== DOCUMENT INGESTION =====================
def ingest_document(file_path: str, filename: str) -> int:
    """Read a file, split it into chunks, store the chunks' embeddings."""
    # Step 1: load the file
    if filename.lower().endswith(".pdf"):
        docs = PyPDFLoader(file_path).load()
    else:
        docs = TextLoader(file_path, encoding="utf-8").load()

    # Step 2: chunking - cut long text into small overlapping pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # each piece is up to 1000 characters
        chunk_overlap=150,  # pieces share 150 characters so no idea is cut in half
    )
    chunks = splitter.split_documents(docs)

    # Step 3: embed every chunk and save it in the vector store
    vectorstore.add_documents(chunks)
    return len(chunks)


# ===================== QUESTION ANSWERING =====================
def answer_question(question: str) -> dict:
    # Step 1: semantic search - find the 4 chunks closest in meaning to the question
    results = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(chunk.page_content for chunk in results)

    # Step 2: contextual generation - give the LLM the question AND those chunks
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    answer = llm.invoke(prompt).content

    return {"answer": answer, "sources": [c.page_content[:200] for c in results]}
