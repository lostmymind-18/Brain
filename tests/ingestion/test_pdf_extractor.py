"""
Integration tests for PdfExtractor.

We generate a synthetic PDF using PyMuPDF itself (already a dependency),
so tests run offline without downloading fixtures.
"""
from __future__ import annotations

import pytest
import pymupdf

from bookmind_tutor.ingestion import PdfExtractor, PdfEncryptedError, PdfExtractionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path, pages_content: list[list[tuple[str, int]]]):
    """
    Create a minimal PDF in tmp_path.

    pages_content: list of pages; each page is a list of (text, fontsize) tuples
    representing lines to insert top-to-bottom.
    """
    doc = pymupdf.open()
    for lines in pages_content:
        page = doc.new_page(width=595, height=842)  # A4
        y = 60
        for text, fontsize in lines:
            page.insert_text((50, y), text, fontsize=fontsize)
            y += fontsize + 8
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPdfExtractor:
    def test_single_page_non_empty(self, tmp_path):
        path = _make_pdf(tmp_path, [[("Hello world. This is a test.", 12)]])
        extractor = PdfExtractor()
        pages = extractor.extract(path)

        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert "Hello world" in pages[0].text
        assert len(pages[0].spans) > 0

    def test_multi_page_count(self, tmp_path):
        path = _make_pdf(tmp_path, [
            [("Page one content.", 12)],
            [("Page two content.", 12)],
            [("Page three content.", 12)],
        ])
        pages = PdfExtractor().extract(path)
        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[2].page_number == 3

    def test_page_numbers_are_one_indexed(self, tmp_path):
        path = _make_pdf(tmp_path, [[("text", 12)], [("text", 12)]])
        pages = PdfExtractor().extract(path)
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    def test_spans_carry_font_metadata(self, tmp_path):
        path = _make_pdf(tmp_path, [[("Sample text", 14)]])
        pages = PdfExtractor().extract(path)
        spans = pages[0].spans

        assert spans, "should have at least one span"
        for span in spans:
            assert span.font, "font name should not be empty"
            assert span.size > 0
            assert len(span.bbox) == 4

    def test_large_document_processes_without_error(self, tmp_path):
        """Smoke test for 50-page document (proxy for 300+ page books)."""
        content = [
            [("Chapter " + str(i), 18), ("Body text on page " + str(i) + ".", 12)]
            for i in range(1, 51)
        ]
        path = _make_pdf(tmp_path, content)
        pages = PdfExtractor().extract(path)
        assert len(pages) == 50
        assert all(p.text for p in pages)

    def test_bad_path_raises_extraction_error(self):
        with pytest.raises(PdfExtractionError):
            PdfExtractor().extract("/nonexistent/path/file.pdf")

    def test_encrypted_pdf_raises_encrypted_error(self, tmp_path):
        # Create an encrypted PDF
        doc = pymupdf.open()
        doc.new_page()
        path = tmp_path / "encrypted.pdf"
        doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret")
        doc.close()

        with pytest.raises(PdfEncryptedError):
            PdfExtractor().extract(path)


class TestExtractToc:
    def _make_pdf_with_toc(self, tmp_path, toc_entries: list[list]):
        doc = pymupdf.open()
        for _ in toc_entries or [None]:
            doc.new_page()
        if toc_entries:
            doc.set_toc(toc_entries)
        path = tmp_path / "with_toc.pdf"
        doc.save(str(path))
        doc.close()
        return path

    def test_pdf_with_embedded_toc_returns_entries(self, tmp_path):
        toc = [[1, "Chapter 1", 1], [2, "Section 1.1", 1], [1, "Chapter 2", 2]]
        path = self._make_pdf_with_toc(tmp_path, toc)
        result = PdfExtractor().extract_toc(path)
        assert len(result) == 3
        assert result[0][1] == "Chapter 1"
        assert result[1][0] == 2  # level

    def test_pdf_without_toc_returns_empty_list(self, tmp_path):
        path = _make_pdf(tmp_path, [[("text", 12)]])
        result = PdfExtractor().extract_toc(path)
        assert result == []

    def test_bad_path_raises_extraction_error(self):
        with pytest.raises(PdfExtractionError):
            PdfExtractor().extract_toc("/nonexistent/file.pdf")
