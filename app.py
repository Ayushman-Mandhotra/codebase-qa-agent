"""
Public Gradio demo for the codebase Q&A agent.

Loads the prebuilt chroma_db/ (see scripts/prebuild_index.py) so requests/
click/tqdm answer instantly with no cloning. Visitors can also paste any
other GitHub repo URL — the agent will clone and index it live, which takes
several minutes of rate-limited embedding calls (shown as a visible step in
the tool-call trace, not a frozen UI). Since indexing shares a scarce daily
embedding quota across every visitor, new live-indexing is capped per day
(MAX_NEW_REPOS_PER_DAY) — already pre-indexed repos are unaffected by that cap.
"""
import os
import uuid

import gradio as gr
import spaces

import agents
from tools.repo_tool import REPOS_DIR, clone_repo

# Hugging Face's free tier only offers ZeroGPU hardware for Gradio Spaces
# (no free plain-CPU tier without a PRO subscription), and ZeroGPU refuses to
# start an app with no @spaces.GPU function. This app does no local GPU work
# at all (just Gemini API calls + local Chroma search) — this no-op function,
# called once at startup, exists purely to satisfy that requirement.
@spaces.GPU
def _warm_zerogpu():
    return True


_warm_zerogpu()

MAX_QUESTION_LENGTH = 500
MAX_REQUESTS_PER_SESSION = 20
MAX_NEW_REPOS_PER_DAY = 5

PRE_INDEXED_REPO_URLS = [
    "https://github.com/psf/requests",
    "https://github.com/pallets/click",
    "https://github.com/tqdm/tqdm",
]

# chroma_db/ (the actual indexed vectors) is committed to the repo, but
# _repos/ (the raw cloned source) is gitignored — on a fresh Space container
# only chroma_db/ exists. retrieve_code_tool only needs the vector store, but
# read_file_tool/list_files_tool read the actual files on disk, so re-clone
# the pre-indexed repos here (cheap git-only operation, no embedding calls,
# and clone_repo() is a no-op if a repo is already present).
for _url in PRE_INDEXED_REPO_URLS:
    clone_repo(_url)

AVAILABLE_REPOS = {
    "requests": os.path.join(REPOS_DIR, "requests"),
    "click": os.path.join(REPOS_DIR, "click"),
    "tqdm": os.path.join(REPOS_DIR, "tqdm"),
}

_agent = agents.build_agent(
    allow_live_clone=True,
    available_repos=AVAILABLE_REPOS,
    max_new_repos_per_day=MAX_NEW_REPOS_PER_DAY,
)

TOOL_ICONS = {
    "clone_repo_tool": "☁️",
    "index_repo_tool": "🧮",
    "retrieve_code_tool": "🔎",
    "list_files_tool": "📂",
    "read_file_tool": "📄",
}

INTRO = f"""
# 🧭 Codebase Q&A Agent — demo

Ask a question about one of three pre-indexed open-source Python repos —
**requests**, **click**, or **tqdm** — for an instant answer, or paste any
other public GitHub repo URL and the agent will clone and index it live.
Watch the tool-call trace to see it decide, step by step, which tools to
call (cloning, semantic code search, file listing, file reads) before
answering with file-path citations.

**Live indexing takes several minutes** (rate-limited embedding calls) and,
since that quota is shared across every visitor, is capped at
{MAX_NEW_REPOS_PER_DAY} newly-indexed repos per day — the pre-indexed repos
above are unaffected by that cap and always answer instantly.
"""

EXAMPLE_QUESTIONS = [
    "How does requests manage connection pooling? Cite the file.",
    "Where does click parse command-line options?",
    "How does tqdm calculate ETA for a progress bar?",
    "How does requests decide whether to follow a redirect?",
]
# ChatInterface requires each example to be a list matching [message, *additional_inputs]
# when additional_inputs are configured — the extra Nones are ignored placeholders
# for thread_id_state/request_count_state, which examples shouldn't override.
EXAMPLES = [[q, None, None] for q in EXAMPLE_QUESTIONS]


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content) if content is not None else ""


def _format_args(args: dict) -> str:
    shown = {k: v for k, v in args.items() if k != "repo_path"}
    return ", ".join(f'{k}="{v}"' for k, v in shown.items())


def _render(steps: list[str], final_text: str | None) -> str:
    if final_text is None:
        body = "\n".join(f"- {s}" for s in steps)
        return f"{body}\n\n_thinking..._" if steps else "_thinking..._"
    if not steps:
        return final_text
    trace = "\n".join(f"- {s}" for s in steps)
    return (
        f"<details><summary>🛠️ Tool calls ({len(steps)})</summary>\n\n"
        f"{trace}\n\n</details>\n\n{final_text}"
    )


def chat(message, history, thread_id, request_count):
    request_count = request_count or 0

    if not message or not message.strip():
        yield "Please enter a question.", request_count
        return

    if len(message) > MAX_QUESTION_LENGTH:
        yield (
            f"That question is {len(message)} characters — please keep it under "
            f"{MAX_QUESTION_LENGTH} for this demo.",
            request_count,
        )
        return

    if request_count >= MAX_REQUESTS_PER_SESSION:
        yield (
            f"You've reached this demo's limit of {MAX_REQUESTS_PER_SESSION} "
            "questions per session. Refresh the page to start a new session.",
            request_count,
        )
        return

    request_count += 1
    config = {"configurable": {"thread_id": thread_id}}
    steps: list[str] = []

    yield _render(steps, None), request_count

    try:
        for update in _agent.stream(
            {"messages": [("user", message)]}, config=config, stream_mode="updates"
        ):
            for node_name, node_data in update.items():
                for m in node_data.get("messages", []):
                    if node_name == "agent":
                        tool_calls = getattr(m, "tool_calls", None) or []
                        if tool_calls:
                            for tc in tool_calls:
                                icon = TOOL_ICONS.get(tc["name"], "🔧")
                                arg_str = _format_args(tc.get("args", {}))
                                steps.append(f"{icon} calling `{tc['name']}`({arg_str})")
                                if tc["name"] == "index_repo_tool":
                                    steps.append(
                                        "⏳ indexing a new repo takes several minutes "
                                        "(rate-limited embedding calls) — hang tight..."
                                    )
                            yield _render(steps, None), request_count
                        else:
                            final_text = _extract_text(getattr(m, "content", ""))
                            yield _render(steps, final_text), request_count
                    elif node_name == "tools":
                        name = getattr(m, "name", "tool")
                        icon = TOOL_ICONS.get(name, "🔧")
                        steps.append(f"{icon} `{name}` → done")
                        yield _render(steps, None), request_count
    except Exception:
        import traceback
        traceback.print_exc()  # full traceback goes to the Space's container logs
        yield (
            _render(steps, None).replace("_thinking..._", "")
            + "\n\nSomething went wrong answering that — this has been logged. "
            "Please try again or ask something else.",
            request_count,
        )


with gr.Blocks(title="Codebase Q&A Agent") as demo:
    gr.Markdown(INTRO)
    thread_id_state = gr.State(lambda: str(uuid.uuid4()))
    request_count_state = gr.State(0)

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[thread_id_state, request_count_state],
        additional_outputs=[request_count_state],
        examples=EXAMPLES,
    )

if __name__ == "__main__":
    demo.launch()
