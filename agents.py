from datetime import date

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools.repo_tool import clone_repo, list_repo_files, read_file
from tools.vectorstore import index_repo, retrieve, is_indexed

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


@tool
def clone_repo_tool(repo_url: str) -> str:
    """Clone a git repo by its URL so its code becomes available to the other tools. Call this first."""
    repo_path = clone_repo(repo_url)
    return f"Repo cloned to local path: {repo_path}. Pass this exact path to the other tools."

@tool
def index_repo_tool(repo_path: str) -> str:
    """Index a cloned repo (its local path, from clone_repo_tool) so its code becomes searchable. Call this before searching."""
    try:
        if is_indexed(repo_path):
            return "Already indexed — no need to index again."
        index_repo(repo_path)
    except Exception as e:
        return (
            f"Indexing failed ({type(e).__name__}). This is usually a temporary rate "
            "limit or daily quota issue with the embeddings API — wait a bit and try again."
        )
    return "Repo indexed successfully. You can now search it."

def _make_limited_index_tool(max_new_repos_per_day: int):
    """
    Builds an index_repo_tool variant that caps how many *new* repos it will
    index per calendar day (already-indexed repos are always free to re-check).

    This exists because embedding calls share one free-tier daily quota
    (1000/day) across indexing AND every retrieve_code_tool query — without
    a cap, a couple of visitors live-indexing different repos on a public
    demo could exhaust that quota for everyone else for the rest of the day.
    State is in-memory and resets both on process restart and at UTC midnight.
    """
    state = {"date": None, "count": 0, "seen": set()}

    @tool
    def index_repo_tool(repo_path: str) -> str:
        """Index a cloned repo (its local path, from clone_repo_tool) so its code becomes searchable. Call this before searching."""
        try:
            if is_indexed(repo_path):
                return "Already indexed — no need to index again."

            today = date.today()
            if state["date"] != today:
                state["date"] = today
                state["count"] = 0
                state["seen"] = set()
            if repo_path not in state["seen"]:
                if state["count"] >= max_new_repos_per_day:
                    return (
                        f"This demo can index at most {max_new_repos_per_day} new repos per "
                        "day, to protect a shared embeddings quota. That limit has been reached "
                        "for today — please try again tomorrow, or ask about one of the "
                        "already pre-indexed repos instead."
                    )
                state["count"] += 1
                state["seen"].add(repo_path)

            index_repo(repo_path)
        except Exception as e:
            return (
                f"Indexing failed ({type(e).__name__}). This is usually a temporary rate "
                "limit or daily quota issue with the embeddings API — wait a bit and try again."
            )
        return "Repo indexed successfully. You can now search it."

    return index_repo_tool


@tool
def retrieve_code_tool(query: str, repo_path: str) -> str:
    """Search a specific repo's indexed code (by its local path) for content relevant to a question. Use this before answering anything about the codebase."""
    try:
        results = retrieve(query, repo_path, k=5)
    except Exception as e:
        return (
            f"Search is temporarily unavailable ({type(e).__name__}). This is usually a "
            "temporary rate limit or daily quota issue with the embeddings API — tell the "
            "user to try again in a bit, don't guess at an answer without this context."
        )
    if not results:
        return "Nothing relevant found."
    return "\n\n".join(
        f"[{doc.metadata['file_path']} | chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
        for doc in results
    )

@tool
def list_files_tool(repo_path: str) -> str:
    """List the code files in a repo (by its local path), to understand its structure before deciding what to read."""
    files = list_repo_files(repo_path)
    return "\n".join(files)

@tool
def read_file_tool(repo_path: str, relative_path: str) -> str:
    """Read the full contents of one specific file in a repo, given the repo's local path and the file's path relative to the repo root (e.g. 'src/adapters.py')."""
    return read_file(repo_path, relative_path)


SYSTEM_PROMPT_FULL = """You are a codebase assistant. Given a repo URL and a question:
1. If no repo is cloned yet, call clone_repo_tool with the URL. It returns the repo's local path.
2. Call index_repo_tool with that exact local path so the code becomes searchable.
3. Use retrieve_code_tool with that same local path to find relevant code before answering.
4. If a retrieved chunk isn't enough context, use read_file_tool (with that path) to read that file in full.
5. Use list_files_tool (with that path) if you need to understand the repo's overall structure first.
6. Always pass the exact repo path returned by clone_repo_tool to every other tool call — don't guess it.
7. Always cite the specific file path(s) your answer is based on.
8. If the retrieved code doesn't actually support an answer, say so honestly.
"""

SYSTEM_PROMPT_RESTRICTED = """You are a codebase assistant for a public demo. Live repo cloning is disabled \
here to keep responses fast on free hosting — indexing a new repo takes several minutes of rate-limited \
embedding calls, too slow for a live web request. Only the following repos are pre-indexed and available, \
by their local path:
{repo_list}

Given a question:
1. Identify which pre-indexed repo (by the local path above) the question is about.
2. Use retrieve_code_tool with that repo's exact local path to find relevant code before answering.
3. If a retrieved chunk isn't enough context, use read_file_tool (with that path) to read that file in full.
4. Use list_files_tool (with that path) if you need to understand the repo's overall structure first.
5. Always cite the specific file path(s) your answer is based on.
6. If asked about a repo that isn't in the list above, explain that live cloning is disabled in this demo \
and name the repos that ARE available instead. Do not attempt to clone or index anything.
7. If the retrieved code doesn't actually support an answer, say so honestly.
"""

SYSTEM_PROMPT_HYBRID = """You are a codebase assistant. The following repos are already pre-indexed and \
instantly searchable, by local path:
{repo_list}

If a question is about one of these, skip straight to retrieve_code_tool with that exact path — do NOT \
call clone_repo_tool or index_repo_tool for them, they're already indexed.

For any other repo the user names by URL:
1. Call clone_repo_tool with the URL. It returns the repo's local path.
2. Call index_repo_tool with that exact path so the code becomes searchable. Warn the user this can take \
several minutes (rate-limited embedding calls) — don't imply it's instant, and don't repeat the call while \
waiting.
3. Use retrieve_code_tool with that same path to find relevant code before answering.
4. If a retrieved chunk isn't enough context, use read_file_tool (with that path) to read that file in full.
5. Use list_files_tool (with that path) if you need to understand the repo's overall structure first.
6. Always pass the exact repo path returned by clone_repo_tool to every other tool call — don't guess it.
7. Always cite the specific file path(s) your answer is based on.
8. If the retrieved code doesn't actually support an answer, say so honestly.
9. If index_repo_tool reports a daily indexing limit was reached, tell the user plainly and suggest one of \
the pre-indexed repos above instead — don't retry indexing.
"""


def build_agent(
    allow_live_clone: bool = True,
    available_repos: dict[str, str] | None = None,
    max_new_repos_per_day: int | None = None,
):
    """
    Builds a compiled LangGraph agent.

    allow_live_clone=False structurally removes clone_repo_tool/index_repo_tool
    from the tool list (not just a runtime check) — for a deployment that must
    never trigger live cloning/indexing. Local/CLI use keeps the default
    (allow_live_clone=True).

    available_repos maps a friendly name to a local repo path, used to tell the
    agent which repos are already pre-indexed (searchable instantly, no cloning
    needed) — combinable with allow_live_clone=True for a hybrid agent that
    answers pre-indexed repos instantly but can also clone+index anything else.

    max_new_repos_per_day, when allow_live_clone=True, caps how many *new*
    repos index_repo_tool will index per day (shared embedding quota
    protection for a public deployment) — None means unlimited (the default,
    matching local/CLI use where there's no "shared with other visitors" risk).
    """
    if allow_live_clone:
        index_tool = (
            _make_limited_index_tool(max_new_repos_per_day)
            if max_new_repos_per_day is not None
            else index_repo_tool
        )
        tools = [clone_repo_tool, index_tool, retrieve_code_tool, list_files_tool, read_file_tool]
        if available_repos:
            repo_list = "\n".join(f"- {name}: {path}" for name, path in available_repos.items())
            prompt = SYSTEM_PROMPT_HYBRID.format(repo_list=repo_list)
        else:
            prompt = SYSTEM_PROMPT_FULL
    else:
        tools = [retrieve_code_tool, list_files_tool, read_file_tool]
        repo_list = "\n".join(f"- {name}: {path}" for name, path in (available_repos or {}).items())
        prompt = SYSTEM_PROMPT_RESTRICTED.format(repo_list=repo_list)

    return create_react_agent(llm, tools=tools, prompt=prompt, checkpointer=MemorySaver())


_default_agent = build_agent(allow_live_clone=True)
agent = _default_agent  # kept as a public name for backward compatibility


def ask(question: str, thread_id: str = "default", agent=None) -> str:
    active_agent = agent if agent is not None else _default_agent
    config = {"configurable": {"thread_id": thread_id}}
    result = active_agent.invoke({"messages": [("user", question)]}, config=config)
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content

if __name__ == "__main__":
    answer = ask("Clone https://github.com/psf/requests, then tell me: how does the library handle request timeouts? Cite the file.")
    print(answer)
