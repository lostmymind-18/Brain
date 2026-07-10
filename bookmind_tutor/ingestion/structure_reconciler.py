"""
StructureReconciler: enhances a TOC-based DocumentTree with sub-heading nodes
discovered by the Tier 3 font-size heuristic.

The embedded PDF TOC gives a reliable Part/Chapter skeleton (usually 2 levels).
The font-size heuristic finds more depth but cannot anchor headings correctly
without the TOC as ground truth. This module combines both:

  TOC entries  ->  anchor nodes (the existing tree structure, fixed)
  Heuristic candidates not matched to any TOC entry  ->  injected as L2/L3 children

Key design choice (from plan):
- Never produces a tree worse than the input: if no candidates survive filtering
  or can be located in the text, the anchor_tree is returned unchanged.
- Local font-size ranking per anchor, NOT global DBSCAN: prevents misleveling
  across chapters when one chapter uses larger subheading fonts than another.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import DocumentNode, DocumentTree, PageContent
from .utils import normalize_heading

# Maximum pages of leeway when cross-checking candidate page vs. TOC page.
# Needed because the page-offset heuristic can be off by 1-2 pages.
_PAGE_WINDOW = 3


@dataclass
class HeadingCandidate:
    """A heading line detected by Tier 3 font-size heuristic."""
    text: str
    page: int
    font_size: float
    bold: bool
    y_pos: float   # vertical center in PDF pts; smaller = higher on page
    level: int     # heuristic level: 0=chapter-like, 1=section-like, 2=sub-section


class StructureReconciler:
    """
    Attach unmatched heading candidates as sub-heading children of TOC anchor nodes.

    Args:
        min_heading_pages: A heading font size must appear on at least this many
            distinct pages to be considered a real heading level. Reuses the same
            threshold as StructureDetector._build_heading_level_map().
        total_pages: Total page count of the document. Used to compute the
            adaptive effective minimum (same formula as StructureDetector).
    """

    def __init__(self, min_heading_pages: int = 3, total_pages: int = 0) -> None:
        self._min_heading_pages = min_heading_pages
        self._total_pages = total_pages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        toc: list[list],               # [[level, title, page], ...]
        candidates: list[HeadingCandidate],
        anchor_tree: DocumentTree,
        pages: list[PageContent],
    ) -> DocumentTree:
        """
        Return anchor_tree enriched with sub-heading nodes from candidates.

        Falls back to returning anchor_tree unchanged if:
        - no candidates survive the false-positive filter
        - all surviving candidates match TOC entries (already anchored)
        """
        if not candidates or not anchor_tree.root_nodes:
            return anchor_tree

        page_texts: dict[int, str] = {p.page_number: p.text for p in pages}

        # Build the normalized TOC title set for O(1) lookup
        toc_norms: set[str] = {
            normalize_heading(str(e[1])) for e in toc if len(e) >= 2 and e[1]
        }

        # Find the first anchor page to reject pre-book candidates
        # (title page, publisher praise, etc.)
        all_nodes = _flatten_tree(anchor_tree.root_nodes)
        first_anchor_page = min(n.page_range[0] for n in all_nodes)

        # Adaptive minimum page frequency (mirrors StructureDetector logic)
        effective_min = max(1, min(self._min_heading_pages, self._total_pages // 40))

        # Per-font-size page counts for frequency filter
        size_page_map: dict[float, set[int]] = {}
        for c in candidates:
            size_page_map.setdefault(c.font_size, set()).add(c.page)
        frequent_sizes: set[float] = {
            fs for fs, pgs in size_page_map.items() if len(pgs) >= effective_min
        }

        # Filter: remove false positives and TOC-level headings
        norm = normalize_heading
        sub_candidates = [
            c for c in candidates
            if (
                len(norm(c.text)) > 1                  # non-trivial after normalize
                and any(ch.isalpha() for ch in c.text) # has at least one letter
                and c.page >= first_anchor_page         # not before TOC content
                and c.font_size in frequent_sizes       # font appears on enough pages
                and norm(c.text) not in toc_norms       # not a TOC-level heading
            )
        ]

        if not sub_candidates:
            return anchor_tree

        # Inject into each anchor node that directly owns pages with candidates
        for node in all_nodes:
            self._inject_into_node(node, sub_candidates, page_texts)

        return anchor_tree

    # ------------------------------------------------------------------
    # Per-node injection
    # ------------------------------------------------------------------

    def _inject_into_node(
        self,
        node: DocumentNode,
        all_sub_candidates: list[HeadingCandidate],
        page_texts: dict[int, str],
    ) -> None:
        """
        Find candidates within this node's directly-owned pages and inject them
        as new child DocumentNodes, splitting the node's raw_text accordingly.
        """
        own_pages = set(self._get_own_pages(node))
        if not own_pages:
            return

        mine = [c for c in all_sub_candidates if c.page in own_pages]
        if not mine:
            return

        # Local font-size ranking: largest = node.level+1, next = node.level+2, ...
        # Capped at node.level+2 to avoid manufacturing unrealistic depth.
        local_sizes = sorted({c.font_size for c in mine}, reverse=True)
        size_to_level: dict[float, int] = {
            size: min(node.level + 1 + i, node.level + 2)
            for i, size in enumerate(local_sizes)
        }

        # Sort by reading order
        mine.sort(key=lambda c: (c.page, c.y_pos))

        new_children, new_raw = self._split_raw_text(
            node.raw_text, mine, size_to_level, node, own_pages, page_texts,
        )

        if new_children:
            node.raw_text = new_raw
            all_children = node.children + new_children
            all_children.sort(key=lambda n: n.page_range[0])
            node.children = all_children

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    def _split_raw_text(
        self,
        raw_text: str,
        candidates: list[HeadingCandidate],
        size_to_level: dict[float, int],
        parent: DocumentNode,
        own_pages: set[int],
        page_texts: dict[int, str],
    ) -> tuple[list[DocumentNode], str]:
        """
        Walk through raw_text line by line, splitting at each candidate heading.

        Returns (new_child_nodes, updated_parent_raw_text).
        """
        if not raw_text.strip():
            return [], raw_text

        new_children: list[DocumentNode] = []
        # Text accumulator for the current scope (parent or last child)
        current_parts: list[str] = []
        parent_text_parts: list[str] = []

        # Map from heading line (normalized) to the matching candidate
        cand_by_norm: dict[str, HeadingCandidate] = {}
        for c in candidates:
            key = normalize_heading(c.text)
            if key and key not in cand_by_norm:
                cand_by_norm[key] = c

        for line in raw_text.split("\n"):
            norm_line = normalize_heading(line)
            if norm_line in cand_by_norm:
                cand = cand_by_norm[norm_line]
                accumulated = "\n".join(current_parts).strip()

                if new_children:
                    # Flush accumulated text into the most recent child
                    existing = new_children[-1].raw_text
                    new_children[-1].raw_text = (
                        (existing.rstrip() + "\n\n" + accumulated).strip()
                        if existing.strip()
                        else accumulated
                    )
                else:
                    # Flush into parent
                    parent_text_parts.append(accumulated)

                current_parts = []

                # Determine end page
                idx = candidates.index(cand)
                end_page = (
                    candidates[idx + 1].page - 1
                    if idx + 1 < len(candidates)
                    else parent.page_range[1]
                )
                end_page = max(cand.page, end_page)

                child = DocumentNode(
                    title=cand.text,
                    level=size_to_level.get(cand.font_size, parent.level + 1),
                    page_range=(cand.page, end_page),
                    children=[],
                    raw_text="",
                )
                new_children.append(child)
                # Remove from lookup so the same heading isn't matched twice
                del cand_by_norm[norm_line]
            else:
                current_parts.append(line)

        # Flush remaining text
        remaining = "\n".join(current_parts).strip()
        if new_children:
            existing = new_children[-1].raw_text
            new_children[-1].raw_text = (
                (existing.rstrip() + "\n\n" + remaining).strip()
                if existing.strip()
                else remaining
            )
        else:
            parent_text_parts.append(remaining)

        updated_parent_raw = "\n\n".join(p for p in parent_text_parts if p.strip())
        return new_children, updated_parent_raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_own_pages(node: DocumentNode) -> list[int]:
        """Pages directly owned by this node (not covered by any child's range)."""
        child_pages: set[int] = set()
        for child in node.children:
            child_pages.update(range(child.page_range[0], child.page_range[1] + 1))
        return sorted(
            p for p in range(node.page_range[0], node.page_range[1] + 1)
            if p not in child_pages
        )


def _flatten_tree(root_nodes: list[DocumentNode]) -> list[DocumentNode]:
    """DFS traversal returning all nodes in the tree."""
    result: list[DocumentNode] = []

    def _dfs(node: DocumentNode) -> None:
        result.append(node)
        for child in node.children:
            _dfs(child)

    for node in root_nodes:
        _dfs(node)
    return result
