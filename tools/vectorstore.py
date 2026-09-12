import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .repo_tool import list_repo_files, read_file
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_splitter_for(file_extension: str) -> RecursiveCharacterTextSplitter:

    """
    Returns the right text splitter for a given file type.

    A naive splitter just cuts every N characters, which can slice a
    function in half and produce a nonsensical chunk. For languages we
    recognize, we use a splitter that understands that language's syntax
    well enough to prefer cutting at function/class boundaries instead.
    Anything we don't recognize (.md, .json, etc.) falls back to the
    plain generic splitter.
    """

    language_map = {".py": Language.PYTHON, ".js": Language.JS, ".ts": Language.TS}
    language = language_map.get(file_extension)
    if language:
        return RecursiveCharacterTextSplitter.from_language(language, chunk_size=1000, chunk_overlap=150)
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

def index_repo(repo_path: str, batch_size: int = 10, delay_seconds: float = 10) -> Chroma:
    """
    Walks every code file in a cloned repo, splits each into chunks,
    embeds them, and stores them in a local Chroma vector store.

    Sends chunks to the embeddings API in small batches with a pause
    between each, instead of all at once — Google's free tier only
    allows 100 embedding requests per minute, and indexing a whole repo
    in one burst can exceed that and get rate-limited partway through.
    """
    store = Chroma(collection_name="repo_code", embedding_function=embeddings, persist_directory="chroma_db")
    docs = []
    for rel_path in list_repo_files(repo_path):
        content = read_file(repo_path, rel_path)
        if not content.strip():
            continue
        ext = os.path.splitext(rel_path)[1]
        chunks = get_splitter_for(ext).split_text(content)
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"file_path": rel_path, "chunk_index": i}))

    # Send docs in small batches with a pause between each, so we stay
    # well under the free tier's 100-requests-per-minute limit instead
    # of bursting everything at once and getting rate-limited.
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        store.add_documents(batch)
        print(f"Indexed {min(i + batch_size, len(docs))}/{len(docs)} chunks...")
        if i + batch_size < len(docs):
            time.sleep(delay_seconds)

    return store

def retrieve(query: str, k: int = 5) -> list[Document]:
    """
    Given a question, finds the k most relevant chunks already indexed.
    """
    store = Chroma(collection_name="repo_code", embedding_function=embeddings, persist_directory="chroma_db")
    return store.similarity_search(query, k=k)

def is_indexed(repo_path: str) -> bool:
    """Cheap check: has this repo already been indexed?"""
    store = Chroma(collection_name="repo_code", embedding_function=embeddings, persist_directory="chroma_db")
    return len(store.similarity_search("x", k=1)) > 0

if __name__ == "__main__":
    index_repo("_repos/requests")
    print("Indexing done.")

    results = retrieve("how does the library handle request timeouts?")
    for r in results:
        print(f"\n[{r.metadata['file_path']} | chunk {r.metadata['chunk_index']}]")
        print(r.page_content[:200])