"""
AST-based code chunker.

Why this exists:
Naive text chunking (e.g. "split every 500 characters") frequently cuts a
function in half, or separates a function from its docstring/decorators.
This destroys retrieval quality for code, because the embedded chunk no
longer represents a coherent unit of meaning.

This chunker parses Python source with the `ast` module and emits one chunk
per top-level function / class / method, preserving:
  - the function/class signature
  - decorators
  - docstring
  - full body
  - source file path + line range (used later for citations)

If a single function is enormous (rare, but happens), it still emits the
whole function as one chunk rather than splitting it — code chunks should
stay semantically whole. Very large functions can be flagged separately by
the eval harness rather than silently mangled here.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


DUPLICATE_SIMILARITY_THRESHOLD = 0.75
DUPLICATE_GROUP_MIN_SIZE = 3


def _cluster_by_docstring_similarity(nodes: list, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> list[list]:
    """
    Groups AST nodes whose docstrings are near-duplicates of each other
    (e.g. FastAPI's get/post/put/patch/... methods, which share one
    docstring template differing only in the HTTP verb word).
    Nodes without a docstring are never clustered -- returned as
    singleton groups.
    Uses union-find over pairwise difflib similarity; simple and fast
    enough for the small number of methods any one class has.
    """
    n = len(nodes)
    docstrings = [ast.get_docstring(node) for node in nodes]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if not docstrings[i]:
            continue
        for j in range(i + 1, n):
            if not docstrings[j]:
                continue
            ratio = max(
                SequenceMatcher(None, docstrings[i], docstrings[j], autojunk=False).ratio(),
                SequenceMatcher(None, docstrings[j], docstrings[i], autojunk=False).ratio(),
            )
            if ratio >= threshold:
                union(i, j)

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(nodes[i])
    return list(groups.values())


@dataclass
class CodeChunk:
    file_path: str
    symbol_name: str          # e.g. "solve_dependencies" or "Depends.__init__"
    symbol_type: str          # "function" | "class" | "method"
    start_line: int
    end_line: int
    source: str
    parent_class: str | None = None
    docstring: str | None = None
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # Stable content-based id so re-indexing can detect "did this
        # specific chunk actually change" rather than re-embedding
        # everything in a touched file.
        digest = hashlib.sha256(
            f"{self.file_path}:{self.symbol_name}:{self.source}".encode()
        ).hexdigest()[:16]
        self.chunk_id = f"{self.file_path}::{self.symbol_name}::{digest}"

    def to_metadata(self) -> dict:
        # ChromaDB's metadata validator rejects None outright (only str,
        # int, float, bool are allowed) -- so any field that can be None
        # (parent_class, docstring) must be normalized to "" instead.
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_class": self.parent_class or "",
            "chunk_id": self.chunk_id,
            # A clean, truncated copy of just the source code, stored
            # separately from the BM25/embedding text (which mixes in the
            # symbol name and auto-generated summary -- see pipeline.py).
            # This is what the frontend displays when a citation is
            # expanded, so it needs to be pure code, not the retrieval
            # -oriented blob.
            "source_preview": self.source[:1500],
        }


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    """Extract exact source text for a node, including decorators."""
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno
    return "\n".join(source_lines[start - 1 : end])


def chunk_python_file(file_path: str, source_code: str) -> list[CodeChunk]:
    """
    Parse a Python file and return one CodeChunk per top-level function,
    class, and method within a class.
    """
    chunks: list[CodeChunk] = []
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError:
        # Skip files that don't parse (e.g. Python 2 syntax, corrupted files)
        return chunks

    source_lines = source_code.splitlines()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    source=_get_source_segment(source_lines, node),
                    docstring=ast.get_docstring(node),
                )
            )

        elif isinstance(node, ast.ClassDef):
            # Emit the class itself as a chunk (signature + docstring +
            # class-level body), then each method as its own chunk.
            class_header_end = node.body[0].lineno - 1 if node.body else node.lineno
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    source=_get_source_segment(source_lines, node),
                    docstring=ast.get_docstring(node),
                )
            )

            method_nodes = [
                sub for sub in ast.iter_child_nodes(node)
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            # Cluster methods with near-duplicate docstrings (see
            # _cluster_by_docstring_similarity docstring for why -- this
            # targets classes like FastAPI's get/post/put/patch/... which
            # are long individually but crowd retrieval as a set because
            # they're near-identical in content).
            clusters = _cluster_by_docstring_similarity(method_nodes)

            for cluster in clusters:
                if len(cluster) >= DUPLICATE_GROUP_MIN_SIZE:
                    combined_source = "\n\n".join(_get_source_segment(source_lines, s) for s in cluster)
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            symbol_name=f"{node.name}.<{len(cluster)}_similar_methods: {', '.join(s.name for s in cluster)}>",
                            symbol_type="method_group",
                            start_line=min(s.lineno for s in cluster),
                            end_line=max(s.end_lineno for s in cluster),
                            source=combined_source,
                            parent_class=node.name,
                        )
                    )
                else:
                    for sub in cluster:
                        chunks.append(
                            CodeChunk(
                                file_path=file_path,
                                symbol_name=f"{node.name}.{sub.name}",
                                symbol_type="method",
                                start_line=sub.lineno,
                                end_line=sub.end_lineno,
                                source=_get_source_segment(source_lines, sub),
                                parent_class=node.name,
                                docstring=ast.get_docstring(sub),
                            )
                        )

    return chunks


def chunk_python_repo(root_dir: str, ignore_dirs: set[str] | None = None) -> list[CodeChunk]:
    """Walk a repo directory and chunk every .py file found."""
    ignore_dirs = ignore_dirs or {".git", "__pycache__", "venv", ".venv", "node_modules", "tests"}
    all_chunks: list[CodeChunk] = []
    root = Path(root_dir)

    for py_file in root.rglob("*.py"):
        if any(part in ignore_dirs for part in py_file.parts):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel_path = str(py_file.relative_to(root))
        all_chunks.extend(chunk_python_file(rel_path, source))

    return all_chunks


if __name__ == "__main__":
    # quick self-test against this very file
    sample = Path(__file__).read_text()
    result = chunk_python_file("code_chunker.py", sample)
    for c in result:
        print(f"[{c.symbol_type}] {c.symbol_name}  (lines {c.start_line}-{c.end_line})  id={c.chunk_id[:40]}...")
