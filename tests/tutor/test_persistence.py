"""Tests for LibraryStore (persistence.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bookmind_tutor.tutor.persistence import BookRecord, LibraryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> LibraryStore:
    return LibraryStore(tmp_path)


def _make_record(name: str = "Test Book", n_chunks: int = 100) -> BookRecord:
    return BookRecord(name=name, n_chunks=n_chunks, indexed_at="2026-01-01T00:00:00+00:00")


# ---------------------------------------------------------------------------
# Library round-trip
# ---------------------------------------------------------------------------

def test_library_round_trip(store: LibraryStore) -> None:
    books = {
        "test_book": _make_record("Test Book", n_chunks=100),
        "other_book": _make_record("Other Book", n_chunks=250),
    }
    store.save_library(books)
    loaded = store.load_library()

    assert set(loaded) == set(books)
    assert loaded["test_book"].name == "Test Book"
    assert loaded["test_book"].n_chunks == 100
    assert loaded["other_book"].n_chunks == 250


def test_library_empty_on_missing_file(store: LibraryStore) -> None:
    assert store.load_library() == {}


def test_library_empty_on_corrupt_json(store: LibraryStore, tmp_path: Path) -> None:
    (tmp_path / "library.json").write_text("not valid json", encoding="utf-8")
    assert store.load_library() == {}


def test_library_empty_on_version_mismatch(store: LibraryStore, tmp_path: Path) -> None:
    data = {"version": 99, "books": {"x": {"name": "X", "n_chunks": 1, "indexed_at": "t"}}}
    (tmp_path / "library.json").write_text(json.dumps(data), encoding="utf-8")
    assert store.load_library() == {}


def test_library_skips_malformed_record(store: LibraryStore, tmp_path: Path) -> None:
    data = {
        "version": 1,
        "books": {
            "good": {"name": "Good", "n_chunks": 1, "indexed_at": "t"},
            "bad": {"wrong_field": "oops"},
        },
    }
    (tmp_path / "library.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = store.load_library()
    assert "good" in loaded
    assert "bad" not in loaded


def test_library_overwrite(store: LibraryStore) -> None:
    store.save_library({"a": _make_record("A")})
    store.save_library({"b": _make_record("B")})
    loaded = store.load_library()
    assert "a" not in loaded
    assert "b" in loaded


# ---------------------------------------------------------------------------
# History round-trip
# ---------------------------------------------------------------------------

_UI_MESSAGES = [
    {"role": "assistant", "content": "Welcome!", "trace": None, "sources": [], "suggestions": [], "msg_id": 0},
    {"role": "user", "content": "What is coupling?"},
    {"role": "assistant", "content": "Coupling is...", "trace": object(), "sources": ["x"], "suggestions": ["y"], "msg_id": 1},
]


def test_history_round_trip(store: LibraryStore) -> None:
    store.save_history("book1", _UI_MESSAGES)
    loaded = store.load_history("book1")

    assert len(loaded) == 3
    assert loaded[0]["role"] == "assistant"
    assert loaded[0]["content"] == "Welcome!"
    assert loaded[1]["role"] == "user"
    assert loaded[1]["content"] == "What is coupling?"


def test_history_runtime_fields_stripped_on_save(store: LibraryStore, tmp_path: Path) -> None:
    store.save_history("book1", _UI_MESSAGES)
    raw = json.loads((tmp_path / "history" / "book1.json").read_text())
    for msg in raw["messages"]:
        assert "trace" not in msg
        assert "sources" not in msg
        assert "suggestions" not in msg


def test_history_defaults_restored_on_load(store: LibraryStore) -> None:
    store.save_history("book1", _UI_MESSAGES)
    loaded = store.load_history("book1")
    for msg in loaded:
        assert "trace" in msg
        assert "sources" in msg
        assert "suggestions" in msg
        assert msg["trace"] is None
        assert msg["sources"] == []
        assert msg["suggestions"] == []


def test_history_msg_id_preserved(store: LibraryStore) -> None:
    store.save_history("book1", _UI_MESSAGES)
    loaded = store.load_history("book1")
    assistant_msgs = [m for m in loaded if m["role"] == "assistant"]
    assert assistant_msgs[0]["msg_id"] == 0
    assert assistant_msgs[1]["msg_id"] == 1


def test_history_empty_on_missing_file(store: LibraryStore) -> None:
    assert store.load_history("nonexistent") == []


def test_history_empty_on_corrupt_json(store: LibraryStore, tmp_path: Path) -> None:
    hist_dir = tmp_path / "history"
    hist_dir.mkdir(exist_ok=True)
    (hist_dir / "book1.json").write_text("{bad json", encoding="utf-8")
    assert store.load_history("book1") == []


def test_history_empty_on_version_mismatch(store: LibraryStore, tmp_path: Path) -> None:
    hist_dir = tmp_path / "history"
    hist_dir.mkdir(exist_ok=True)
    data = {"version": 99, "messages": [{"role": "user", "content": "hi"}]}
    (hist_dir / "book1.json").write_text(json.dumps(data), encoding="utf-8")
    assert store.load_history("book1") == []


def test_history_delete(store: LibraryStore, tmp_path: Path) -> None:
    store.save_history("book1", _UI_MESSAGES)
    assert (tmp_path / "history" / "book1.json").exists()
    store.delete_history("book1")
    assert not (tmp_path / "history" / "book1.json").exists()


def test_history_delete_missing_is_silent(store: LibraryStore) -> None:
    store.delete_history("does_not_exist")  # must not raise


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_atomic_write_creates_no_tmp_on_success(store: LibraryStore, tmp_path: Path) -> None:
    store.save_library({"a": _make_record()})
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
