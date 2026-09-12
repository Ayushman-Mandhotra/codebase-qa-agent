import os
import re
import time
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .repo_tool import list_repo_files, read_file
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Absolute path, so the prebuild script and the running app agree on the
# same on-disk location regardless of the process's launch CWD.
CHROMA_DIR = os.environ.get("CHROMA_DIR") or str(Path(__file__).resolve().parent.parent / "chroma_db")


def _collection_name(repo_path: str) -> str:
    """
    Derives a stable Chroma collection name from a repo's local directory
    name, so each repo gets its own collection instead of every repo
    sharing one hardcoded collection (which used to mix unrelated repos'
    chunks together and made is_indexed() report true for the wrong repo).
    """
    name = os.path.basename(os.path.normpath(repo_path))
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = f"repo_{name}".strip("_-")
    return name[:63]


def _store_for(repo_path: str) -> Chroma:
    return Chroma(
        collection_name=_collection_name(repo_path),
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

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
    store = _store_for(repo_path)
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

def retrieve(query: str, repo_path: str, k: int = 5) -> list[Document]:
    """
    Given a question and the local path of the repo it's about, finds the
    k most relevant chunks already indexed for that specific repo.
    """
    store = _store_for(repo_path)
    return store.similarity_search(query, k=k)

def is_indexed(repo_path: str) -> bool:
    """
    Cheap check: has this specific repo already been indexed?

    Uses the collection's raw document count instead of a similarity search,
    so checking status never costs an embedding API call (embedding quota is
    scarce enough — free tier is 1000/day, shared with actual indexing and
    every retrieve() call — that spending it just to check status is wasteful).
    """
    store = _store_for(repo_path)
    return store._collection.count() > 0

if __name__ == "__main__":
    index_repo("_repos/requests")
    print("Indexing done.")

    results = retrieve("how does the library handle request timeouts?", "_repos/requests")
    for r in results:
        print(f"\n[{r.metadata['file_path']} | chunk {r.metadata['chunk_index']}]")
        print(r.page_content[:200])