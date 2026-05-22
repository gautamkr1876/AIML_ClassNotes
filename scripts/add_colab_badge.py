#!/usr/bin/env python3
"""Insert an "Open in Colab" badge as the first markdown cell of each notebook.

Usage:
    python3 scripts/add_colab_badge.py                  # process every .ipynb in the repo
    python3 scripts/add_colab_badge.py path/to/nb.ipynb # process specific files
    python3 scripts/add_colab_badge.py "Data Foundation"  # process all .ipynb under a folder

Idempotent: notebooks that already contain a Colab badge are skipped.
"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

GH_USER = "gautamkr1876"
GH_REPO = "AIML_ClassNotes"
BRANCH = "main"
BADGE_MARKER = "colab-badge.svg"  # idempotency check

REPO_ROOT = Path(__file__).resolve().parent.parent


def encode_path(p: str) -> str:
    """Encode each path segment, mirroring GitHub's URL style (spaces → %20, parens kept)."""
    return "/".join(quote(seg, safe="()._-") for seg in p.split("/"))


def make_badge_cell(rel_path: str) -> dict:
    enc = encode_path(rel_path)
    url = f"https://colab.research.google.com/github/{GH_USER}/{GH_REPO}/blob/{BRANCH}/{enc}"
    md = (
        f'<a href="{url}" target="_blank">\n'
        f'  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>\n'
        f'</a>'
    )
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": md.splitlines(keepends=True),
    }


def already_has_badge(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        if BADGE_MARKER in "".join(cell.get("source", [])):
            return True
    return False


def process(nb_path: Path) -> bool:
    rel = nb_path.relative_to(REPO_ROOT).as_posix()
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    if already_has_badge(nb):
        print(f"SKIP (badge exists): {rel}")
        return False
    nb.setdefault("cells", []).insert(0, make_badge_cell(rel))
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"ADDED: {rel}")
    return True


def discover_notebooks(targets: list[str]) -> list[Path]:
    if not targets:
        roots = [REPO_ROOT]
    else:
        roots = [Path(t) if os.path.isabs(t) else REPO_ROOT / t for t in targets]
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".ipynb":
            found.append(root)
            continue
        if not root.exists():
            print(f"WARN: path not found: {root}", file=sys.stderr)
            continue
        for p in root.rglob("*.ipynb"):
            if ".ipynb_checkpoints" in p.parts:
                continue
            found.append(p)
    return sorted(set(found))


def main(argv: list[str]) -> int:
    notebooks = discover_notebooks(argv[1:])
    if not notebooks:
        print("No notebooks found.")
        return 0
    changed = sum(process(p) for p in notebooks)
    print(f"\nDone. Modified {changed}/{len(notebooks)} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
