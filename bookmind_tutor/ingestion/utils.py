"""Utility helpers for inspecting ingestion pipeline output."""
from __future__ import annotations

import re

from .models import DocumentNode, DocumentTree


def normalize_heading(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace.

    Used for fuzzy heading matching so that "Chapter 1. Introduction" and
    "Chapter 1: Introduction" and "CHAPTER 1 Introduction" all normalize to
    the same string and can be matched against each other.

    This is the shared normalization used by get_book_section (for tool
    matching) and StructureReconciler (for TOC anchor matching).
    """
    cleaned = re.sub(r"[^\w\s]", " ", text.replace("\xa0", " ").lower())
    return " ".join(cleaned.split())


def print_tree(
    tree: DocumentTree,
    max_depth: int = 2,
    max_children: int = 8,
) -> None:
    """
    Print a human-readable ASCII tree of a DocumentTree.

    Args:
        tree: The tree to display.
        max_depth: How many levels deep to expand (0 = root only).
        max_children: Max children to show per node before truncating.
    """
    if not tree.root_nodes:
        print("(empty tree)")
        return

    for node in tree.root_nodes:
        _print_node(node, depth=0, prefix="", max_depth=max_depth, max_children=max_children)


def _print_node(
    node: DocumentNode,
    depth: int,
    prefix: str,
    max_depth: int,
    max_children: int,
) -> None:
    title = (node.title or "[untitled]")[:60]
    raw_chars = len(node.raw_text.strip())
    n_children = len(node.children)

    print(
        f"{prefix}[L{node.level}] "
        f"p{node.page_range[0]}-{node.page_range[1]}  "
        f"{title!r}"
        f"  ({n_children} ch, {raw_chars} chars)"
    )

    if depth >= max_depth:
        if n_children > 0:
            print(f"{prefix}  ... ({n_children} children not shown)")
        return

    shown = node.children[:max_children]
    child_prefix = prefix + "  "
    for child in shown:
        _print_node(child, depth + 1, child_prefix, max_depth, max_children)

    remaining = n_children - len(shown)
    if remaining > 0:
        print(f"{child_prefix}... and {remaining} more")
