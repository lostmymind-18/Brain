"""
LibraryStore: persists the book library and per-book chat history to disk.

Storage layout (all under a configurable root directory):
  library.json        — book registry: {book_id: BookRecord}
  history/
    {book_id}.json    — one file per book, list of simplified message dicts

Design principles:
- Atomic writes (write .tmp then os.replace) so a crash mid-write never
  corrupts the existing file.
- Fault-tolerant reads: missing or corrupt files return empty state, never raise.
  Worst case equals current behavior (session starts fresh).
- Persisted message dicts only keep (role, content, msg_id). Runtime fields
  (trace, sources, suggestions) are stripped on write and re-added with safe
  defaults on read — this keeps files stable even if UI message fields evolve.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_LIBRARY_FILENAME = "library.json"
_HISTORY_DIRNAME = "history"
_SCHEMA_VERSION = 1

# Fields kept when persisting each message
_PERSIST_FIELDS = {"role", "content", "msg_id"}
# Defaults restored on load (runtime-only fields)
_MESSAGE_DEFAULTS: dict = {"trace": None, "sources": [], "suggestions": []}


@dataclass
class BookRecord:
    """Metadata for one indexed book, stored in library.json."""
    name: str
    n_chunks: int
    indexed_at: str  # ISO 8601 timestamp


class LibraryStore:
    """
    Reads and writes the book library and per-book chat history.

    All writes are atomic via a .tmp sibling + os.replace.
    All reads are fault-tolerant: missing/corrupt files return empty state.

    Args:
        root: Directory that holds library.json and history/. Created if absent.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._library_path = self._root / _LIBRARY_FILENAME
        self._history_dir = self._root / _HISTORY_DIRNAME
        self._root.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Library (book registry)
    # ------------------------------------------------------------------

    def load_library(self) -> dict[str, BookRecord]:
        """
        Return {book_id: BookRecord} for all persisted books.

        Returns an empty dict on missing file, version mismatch, or parse error.
        """
        try:
            raw = json.loads(self._library_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Could not read library.json: %s", exc)
            return {}

        if raw.get("version") != _SCHEMA_VERSION:
            logger.warning(
                "library.json schema version %s != expected %s; treating as empty",
                raw.get("version"),
                _SCHEMA_VERSION,
            )
            return {}

        records: dict[str, BookRecord] = {}
        for bid, entry in raw.get("books", {}).items():
            try:
                records[bid] = BookRecord(**entry)
            except TypeError as exc:
                logger.warning("Skipping malformed book record %r: %s", bid, exc)
        return records

    def save_library(self, books: dict[str, BookRecord]) -> None:
        """Atomically persist the full book registry."""
        payload = {
            "version": _SCHEMA_VERSION,
            "books": {bid: asdict(rec) for bid, rec in books.items()},
        }
        self._atomic_write(self._library_path, payload)

    # ------------------------------------------------------------------
    # History (per-book message list)
    # ------------------------------------------------------------------

    def load_history(self, book_id: str) -> list[dict]:
        """
        Return the saved message list for book_id.

        Returns an empty list on missing file, version mismatch, or parse error.
        Runtime fields (trace, sources, suggestions) are re-added with safe defaults.
        """
        path = self._history_dir / f"{book_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.warning("Could not read history for %r: %s", book_id, exc)
            return []

        if raw.get("version") != _SCHEMA_VERSION:
            logger.warning(
                "history/%s.json schema version mismatch; treating as empty", book_id
            )
            return []

        messages = []
        for m in raw.get("messages", []):
            messages.append({**_MESSAGE_DEFAULTS, **m})
        return messages

    def save_history(self, book_id: str, messages: list[dict]) -> None:
        """
        Atomically persist the message list for book_id.

        Only (role, content, msg_id) are kept; runtime fields are stripped
        so the file format stays stable regardless of UI changes.
        """
        slim = [
            {k: m[k] for k in _PERSIST_FIELDS if k in m}
            for m in messages
        ]
        payload = {"version": _SCHEMA_VERSION, "messages": slim}
        self._atomic_write(self._history_dir / f"{book_id}.json", payload)

    def delete_history(self, book_id: str) -> None:
        """Remove the history file for book_id. Silent if already absent."""
        path = self._history_dir / f"{book_id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Could not delete history for %r: %s", book_id, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, path: Path, data: dict) -> None:
        """
        Write JSON to path atomically.

        Writes to a .tmp sibling first, then renames. A crash between the two
        operations leaves the original file intact.
        """
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception as exc:
            logger.error("Failed to write %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
