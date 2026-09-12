"""
Clones and indexes a fixed set of small, well-known Python repos into the
local Chroma store, so the deployed demo app never has to index anything
live (indexing is minutes of rate-limited embedding calls — far too slow
for a web request).

Run this locally, then commit the resulting chroma_db/ directory:

    python scripts/prebuild_index.py

Safe to re-run: clone_repo() skips repos already cloned, and index_repo()
is skipped per-repo if is_indexed() already reports that repo as indexed.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.repo_tool import clone_repo, REPOS_DIR
from tools.vectorstore import index_repo, is_indexed, CHROMA_DIR

# Small, well-known, permissively-licensed pure-Python repos — enough
# variety (HTTP client, CLI framework, progress bars) to make retrieval
# and citation demo-worthy without taking hours to embed.
REPOS = [
    "https://github.com/psf/requests",
    "https://github.com/pallets/click",
    "https://github.com/tqdm/tqdm",
]


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def main() -> None:
    print(f"Repos dir:  {REPOS_DIR}")
    print(f"Chroma dir: {CHROMA_DIR}")
    print("-" * 60)

    for n, url in enumerate(REPOS, start=1):
        name = url.rstrip("/").split("/")[-1]
        print(f"[{n}/{len(REPOS)}] {name}")

        start = time.time()
        repo_path = clone_repo(url)
        print(f"  cloned to {repo_path} ({time.time() - start:.1f}s)")

        try:
            if is_indexed(repo_path):
                print(f"  already indexed — skipping")
                continue

            print(f"  indexing (this takes a few minutes, rate-limited)...")
            start = time.time()
            index_repo(repo_path)
            print(f"  indexed {name} in {time.time() - start:.1f}s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(
                "  Stopping here — this is usually the embeddings API's daily quota "
                "being exhausted (shared across cloning, indexing, and every retrieve "
                "call). Re-run this script later; already-indexed repos are skipped."
            )
            break

    print("-" * 60)
    if os.path.isdir(CHROMA_DIR):
        print(f"chroma_db size on disk: {_dir_size_mb(CHROMA_DIR):.1f} MB")
    print("Done. Commit the chroma_db/ directory to deploy with this index.")


if __name__ == "__main__":
    main()
