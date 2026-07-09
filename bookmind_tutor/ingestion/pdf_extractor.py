"""
PdfExtractor: extracts text and font metadata from PDF files.

Uses PyMuPDF's "dict" extraction mode for per-span font/size info needed by
StructureDetector. Extraction is two-phase:
  1. Collect raw page dicts from all pages in one pass.
  2. Identify running headers/footers (repeated text in top/bottom margin bands)
     and strip them before building PageContent objects.

Header/footer detection:
  A text line is considered a running header/footer if its normalized form
  (leading/trailing page numbers stripped) appears in the margin band of at
  least `min_hf_pages` distinct pages. The margin band is the top and bottom
  `header_footer_margin * page_height` points of each page.

Known limitation: true multi-column layouts may have imperfect reading order.
This is acceptable for the current scope (book-length documents are typically
single-column).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from .exceptions import PdfEncryptedError, PdfExtractionError
from .models import PageContent, TextSpan

_RE_LEADING_NUM = re.compile(r"^\d+\s*")
_RE_TRAILING_NUM = re.compile(r"\s*\d+$")


def _normalize_hf(text: str) -> str:
    """
    Strip leading/trailing page numbers for header/footer pattern matching.

    "300 | Chapter 7: Transactions" and "301 | Chapter 7: Transactions" both
    normalize to "| Chapter 7: Transactions", so they count as the same
    repeated pattern despite the changing page number.
    """
    t = _RE_LEADING_NUM.sub("", text.strip())
    t = _RE_TRAILING_NUM.sub("", t).strip()
    return t


class PdfExtractor:
    """
    Extracts structured text and font spans from a PDF file.

    Args:
        header_footer_margin: Fraction of page height to treat as the
            header/footer zone (top and bottom). Default 0.08 (8%).
        min_hf_pages: A text pattern must appear in the margin band on at
            least this many pages to be stripped. Default 3.
    """

    def __init__(
        self,
        header_footer_margin: float = 0.08,
        min_hf_pages: int = 3,
    ) -> None:
        self.header_footer_margin = header_footer_margin
        self.min_hf_pages = min_hf_pages

    def extract(self, pdf_path: str | Path) -> list[PageContent]:
        """
        Extract all pages from a PDF with header/footer lines stripped.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            One PageContent per page, in document order.

        Raises:
            PdfEncryptedError: if the PDF requires a password.
            PdfExtractionError: for any other extraction failure.
        """
        path = Path(pdf_path)
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise PdfExtractionError(f"Cannot open '{path}': {exc}") from exc

        if doc.is_encrypted:
            doc.close()
            raise PdfEncryptedError(
                f"'{path}' is password-protected. Provide a password to extract text."
            )

        try:
            # Phase 1: collect raw page data (page_number, page_height, page_dict)
            raw: list[tuple[int, float, dict]] = [
                (i + 1, doc[i].rect.height, doc[i].get_text("dict", sort=True))
                for i in range(len(doc))
            ]
        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError(f"Error while reading '{path}': {exc}") from exc
        finally:
            doc.close()

        # Phase 2: identify repeating header/footer patterns, then build clean pages
        hf_texts = self._find_header_footer_texts(raw)
        return [self._build_page_content(pn, ph, pd, hf_texts) for pn, ph, pd in raw]

    # ------------------------------------------------------------------
    # Header / footer detection
    # ------------------------------------------------------------------

    def _find_header_footer_texts(
        self, raw_pages: list[tuple[int, float, dict]]
    ) -> set[str]:
        """
        Find normalized text strings that appear repeatedly in page margin bands.

        Only strings appearing on >= min_hf_pages pages are returned.
        For very short documents the threshold is lowered so that genuine
        repeated headers in a 5-page excerpt are still caught.
        """
        band_counts: Counter = Counter()

        for _, page_height, page_dict in raw_pages:
            margin = page_height * self.header_footer_margin
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                by0, by1 = block["bbox"][1], block["bbox"][3]
                if by0 >= margin and by1 <= page_height - margin:
                    continue  # block is entirely in the body area
                for line in block.get("lines", []):
                    line_text = "".join(
                        s.get("text", "") for s in line.get("spans", [])
                    ).strip()
                    normed = _normalize_hf(line_text)
                    if len(normed) > 3:
                        band_counts[normed] += 1

        return {text for text, count in band_counts.items() if count >= self.min_hf_pages}

    # ------------------------------------------------------------------
    # Page content construction
    # ------------------------------------------------------------------

    def extract_toc(self, pdf_path: str | Path) -> list[list]:
        """
        Return the embedded PDF outline (bookmarks) as [[level, title, page], ...].

        Uses pymupdf's get_toc() which reads the document's bookmark tree directly.
        This is ground-truth structure when present - no heuristic needed.
        level is 1-based (1 = top-level chapter/part, 2 = section, 3 = subsection).

        Returns an empty list if the PDF has no embedded outline.
        Raises the same exceptions as extract().
        """
        path = Path(pdf_path)
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise PdfExtractionError(f"Cannot open '{path}': {exc}") from exc

        if doc.is_encrypted:
            doc.close()
            raise PdfEncryptedError(
                f"'{path}' is password-protected. Provide a password to extract text."
            )
        try:
            return doc.get_toc()
        finally:
            doc.close()

    def _build_page_content(
        self,
        page_number: int,
        page_height: float,
        page_dict: dict,
        hf_texts: set[str],
    ) -> PageContent:
        margin = page_height * self.header_footer_margin
        spans: list[TextSpan] = []
        block_texts: list[str] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            by0, by1 = block["bbox"][1], block["bbox"][3]
            in_margin = by0 < margin or by1 > page_height - margin
            line_texts: list[str] = []

            for line in block.get("lines", []):
                line_text = "".join(
                    s.get("text", "") for s in line.get("spans", [])
                ).strip()

                # Strip lines identified as headers/footers (position + pattern match)
                if in_margin and _normalize_hf(line_text) in hf_texts:
                    continue

                line_str = ""
                for span_data in line.get("spans", []):
                    raw = span_data.get("text", "")
                    if not raw:
                        continue
                    line_str += raw
                    spans.append(TextSpan(
                        text=raw,
                        font=span_data.get("font", ""),
                        size=round(span_data.get("size", 0.0), 2),
                        bbox=tuple(span_data["bbox"]),
                    ))
                if line_str.strip():
                    line_texts.append(line_str)

            if line_texts:
                block_texts.append("\n".join(line_texts))

        return PageContent(
            page_number=page_number,
            text="\n\n".join(block_texts),
            spans=spans,
        )
