# Codebase Q&A Agent

An AI agent that clones a GitHub repository, indexes its code, and answers
questions about how the codebase works — citing the specific file each
answer is grounded in. Built to demonstrate getting productive in an
unfamiliar codebase fast, without a human walking you through it first.

## What "agent" means here

This isn't a single API call. Given a question, the LLM decides — turn by
turn — whether it already has enough grounded context to answer, or
whether it needs to call a tool first: clone the repo, index it,
semantically search it, read a specific file in full, or list the file
tree to orient itself. It loops (tool call → tool result → reasoning)
until it's confident, then answers with file-path citations.

## Architecture

User question
│
▼
LangGraph agent (Gemini + tool calling)
│
├─ clone_repo_tool → git clone --depth 1 <url>
├─ list_files_tool → walks the repo, skips node_modules/.git/etc.
├─ index_repo_tool → language-aware chunking → embeddings → Chroma
├─ retrieve_code_tool → semantic similarity search (RAG)
└─ read_file_tool → full file content when a chunk isn't enough
│
▼
Answer, grounded in retrieved code, with file-path citations


Conversation state persists per `thread_id` via LangGraph's `MemorySaver`,
so a follow-up like "where is that class defined?" resolves without
re-explaining context.

## Key design decisions

**Why Chroma, run locally?** No external vector DB service needed for a
single-repo demo tool — a one-line swap to a managed vector store if this
ever needed to scale to many repos or concurrent users.

**Why a language-aware splitter?** A naive character-count splitter can
cut a function in half. `RecursiveCharacterTextSplitter.from_language`
prefers splitting at function/class boundaries instead.

**Why batch embedding requests with delays?** Google's free tier caps
embedding requests per minute. Indexing a whole repo in one burst can hit
that limit mid-run. Batching with pauses between groups is also just how
production systems handle rate limits generally, not a free-tier-only fix.

**Why both `retrieve_code_tool` and `read_file_tool`?** Retrieval gives
fast, targeted answers for "where is X handled" questions. But some
questions need a whole file's context to answer correctly — giving the
agent both and letting it decide is the actual design choice.

**Read-only, no destructive operations.** The agent only reads code — no
tool can modify, commit, or push anything.

## What I'd add next

- Reranking retrieved chunks before handing them to the LLM.
- Incremental re-indexing when a repo changes, instead of re-indexing from scratch.
- A real code-search index (e.g. tree-sitter based) for repos too large for a flat vector store to handle well.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your real GOOGLE_API_KEY
```

## Run it

```bash
python3 cli.py
```

Ask it to clone and index a repo as your first question, e.g.:
*"Clone https://github.com/psf/requests and explain how sessions work. Cite the file."*
