import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()  
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


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
        persist_directory="./chroma_db", 
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  

def ingest_document(file_path: str, filename: str) -> int:
   
  
    if filename.lower().endswith(".pdf"):
        docs = PyPDFLoader(file_path).load()
    else:
        docs = TextLoader(file_path, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   
        chunk_overlap=150,  
    )
    chunks = splitter.split_documents(docs)


    vectorstore.add_documents(chunks)
    return len(chunks)



def answer_question(question: str) -> dict:
   
    results = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(chunk.page_content for chunk in results)


    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    answer = llm.invoke(prompt).content

    return {"answer": answer, "sources": [c.page_content[:200] for c in results]}
