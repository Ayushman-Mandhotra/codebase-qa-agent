from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools.repo_tool import clone_repo, list_repo_files, read_file
from tools.vectorstore import index_repo, retrieve
from tools.vectorstore import index_repo, retrieve, is_indexed

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

current_repo_path = None

@tool
def clone_repo_tool(repo_url: str) -> str:
    """Clone a git repo by its URL so its code becomes available to the other tools. Call this first."""
    global current_repo_path
    current_repo_path = clone_repo(repo_url)
    return "Repo cloned successfully. You can now index it or list its files."

@tool
def index_repo_tool() -> str:
    """Index the currently cloned repo so its code becomes searchable. Call this after clone_repo_tool, before searching."""
    if current_repo_path is None:
        return "No repo cloned yet — call clone_repo_tool first."
    index_repo(current_repo_path)
    return "Repo indexed successfully. You can now search it."

@tool
def retrieve_code_tool(query: str) -> str:
    """Search the indexed repo for code relevant to a question. Use this before answering anything about the codebase."""
    if current_repo_path is None:
        return "No repo indexed yet."
    results = retrieve(query, k=5)
    if not results:
        return "Nothing relevant found."
    return "\n\n".join(
        f"[{doc.metadata['file_path']} | chunk {doc.metadata['chunk_index']}]\n{doc.page_content}"
        for doc in results
    )

@tool
def list_files_tool() -> str:
    """List the code files in the currently cloned repo, to understand its structure before deciding what to read."""
    if current_repo_path is None:
        return "No repo cloned yet."
    files = list_repo_files(current_repo_path)
    return "\n".join(files)


@tool
def read_file_tool(relative_path: str) -> str:
    """Read the full contents of one specific file, given its path relative to the repo root (e.g. 'src/adapters.py')."""
    if current_repo_path is None:
        return "No repo cloned yet."
    return read_file(current_repo_path, relative_path)

@tool
def index_repo_tool() -> str:
    """Index the currently cloned repo so its code becomes searchable. Call this after clone_repo_tool, before searching."""
    if current_repo_path is None:
        return "No repo cloned yet — call clone_repo_tool first."
    if is_indexed(current_repo_path):
        return "Already indexed — no need to index again."
    index_repo(current_repo_path)
    return "Repo indexed successfully. You can now search it."

system_prompt = """You are a codebase assistant. Given a repo URL and a question:
1. If no repo is cloned yet, call clone_repo_tool with the URL.
2. Call index_repo_tool once so the code becomes searchable.
3. Use retrieve_code_tool to find relevant code before answering.
4. If a retrieved chunk isn't enough context, use read_file_tool to read that file in full.
5. Use list_files_tool if you need to understand the repo's overall structure first.
6. Always cite the specific file path(s) your answer is based on.
7. If the retrieved code doesn't actually support an answer, say so honestly.
"""

checkpointer = MemorySaver()

agent = create_react_agent(
    llm,
    tools=[clone_repo_tool, index_repo_tool, retrieve_code_tool, list_files_tool, read_file_tool],
    prompt=system_prompt,
    checkpointer=checkpointer,
)


def ask(question: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("user", question)]}, config=config)
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content

if __name__ == "__main__":
    answer = ask("Clone https://github.com/psf/requests, then tell me: how does the library handle request timeouts? Cite the file.")
    print(answer)