"""Custom exceptions for the ingestion pipeline."""


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened or read."""


class PdfEncryptedError(PdfExtractionError):
    """Raised specifically for password-protected PDFs."""
