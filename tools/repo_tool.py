import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

# Absolute path, so this resolves the same way regardless of the process's
# current working directory at launch (a relative "_repos" broke when the
# app was started from a different CWD than the repo root).
REPOS_DIR = os.environ.get("REPOS_DIR") or str(Path(__file__).resolve().parent.parent / "_repos")

def clone_repo(url: str) -> str:
    dest = os.path.join(REPOS_DIR, url.rstrip("/").split("/")[-1])
    if os.path.isdir(os.path.join(dest, ".git")):
        return dest  # already cloned, don't do it again
    os.makedirs(REPOS_DIR, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    return dest

SKIP_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build"}
CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".html", ".css"}

def list_repo_files(repo_path: str) -> list[str]:
    results = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in CODE_EXTENSIONS:
                rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                results.append(rel_path)
    return results

def read_file(repo_path: str, rel_path: str) -> str:
    full_path = os.path.join(repo_path, rel_path)
    with open(full_path, "r", errors="ignore") as f:
        return f.read()

if __name__ == "__main__":
    path = clone_repo("https://github.com/psf/requests")
    print("Cloned to:", path)
    files = list_repo_files(path)
    print(f"Found {len(files)} files")
    print(files[:10])
    content = read_file(path, "README.md")
    print(content[:300])