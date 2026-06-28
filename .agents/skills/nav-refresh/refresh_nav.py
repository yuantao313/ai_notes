"""Scan docs/ and update mkdocs.yml nav section."""
import os
import re
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # knowledge_base/
DOCS = ROOT / "docs"
YML = ROOT / "mkdocs.yml"


def extract_title(md_path: Path) -> str:
    """Extract first # heading from a markdown file, fallback to filename."""
    try:
        text = md_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = re.match(r"^#\s+(.+)", line)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    # fallback: filename without ordering prefix like 00_, 01_
    name = md_path.stem
    return re.sub(r"^\d{2,}_", "", name)


def scan_docs() -> OrderedDict:
    """Scan docs/ directory, return {dir_rel: [(rel_path, title), ...]}."""
    groups = OrderedDict()
    root_files = []

    for md_file in sorted(DOCS.rglob("*.md")):
        # skip garbage files like {{...}}.md
        if md_file.name.startswith("{{"):
            continue
        
        title = extract_title(md_file)
        if not title:
            continue

        rel = str(md_file.relative_to(DOCS)).replace("\\", "/")
        parent = str(md_file.parent.relative_to(DOCS))

        if parent == ".":
            root_files.append((rel, title))
        else:
            dir_label = parent.replace("\\", "/")
            groups.setdefault(dir_label, []).append((rel, title))

    # sort: index.md first in each group
    def sort_key(item):
        rel, _ = item
        return (0 if rel.endswith("index.md") else 1, rel)

    root_files.sort(key=sort_key)
    for k in groups:
        groups[k].sort(key=sort_key)

    result = OrderedDict()
    result["."] = root_files
    for k in groups:
        result[k] = groups[k]
    return result


def build_nav_yaml(groups: OrderedDict) -> str:
    """Build nav YAML block from grouped files."""
    lines = ["nav:"]
    indent = "  "

    for dir_label, files in groups.items():
        if dir_label == ".":
            for rel, title in files:
                lines.append(f"{indent}- {title}: {rel}")
        else:
            lines.append(f"{indent}- {dir_label}:")
            for rel, title in files:
                lines.append(f"{indent}  - {title}: {rel}")

    return "\n".join(lines) + "\n"


def update_mkdocs_yml():
    """Replace nav section in mkdocs.yml, preserve everything else."""
    content = YML.read_text(encoding="utf-8")

    groups = scan_docs()
    new_nav = build_nav_yaml(groups)

    # replace existing nav block (from "^nav:" to next top-level key)
    nav_pat = re.compile(r"^nav:.*?(\n(?!\s|$|\n))", re.MULTILINE | re.DOTALL)
    match = nav_pat.search(content)

    if match:
        # insert new nav, keep the trailing newline that separates from next key
        prefix = content[: match.start()]
        suffix = content[match.end() - 1 :]  # keep the separating newline
        new_content = prefix + new_nav.rstrip("\n") + "\n" + suffix
    else:
        # no existing nav, append after site_name
        new_content = content.rstrip("\n") + "\n\n" + new_nav

    YML.write_text(new_content, encoding="utf-8")
    print("mkdocs.yml nav updated.")
    print(new_nav)


if __name__ == "__main__":
    update_mkdocs_yml()
