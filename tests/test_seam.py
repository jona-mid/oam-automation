"""Tests for deadtrees_seam without the platform stack installed."""

from contextlib import contextmanager
from types import SimpleNamespace

import deadtrees_seam


class _FakeQuery:
    def __init__(self, calls, known):
        self.calls = calls
        self.known = known
        self.values = []

    def select(self, _columns):
        return self

    def in_(self, _column, values):
        self.values = list(values)
        return self

    def execute(self):
        self.calls.append(self.values)
        return SimpleNamespace(
            data=[{"sha256": h, "dataset_id": self.known[h]} for h in self.values if h in self.known]
        )


def test_hash_lookup_is_chunked(monkeypatch):
    calls = []
    known = {"h7": 7, "h120": 120}

    @contextmanager
    def use_client(_key):
        yield SimpleNamespace(table=lambda _name: _FakeQuery(calls, known))

    monkeypatch.setenv("SUPABASE_KEY", "anon")
    monkeypatch.setattr(
        deadtrees_seam,
        "_platform_api",
        lambda: (None, use_client, SimpleNamespace(orthos_table="v2_orthos")),
    )
    hashes = [f"h{i}" for i in range(342)]
    result = deadtrees_seam.file_hashes_on_platform(hashes)

    assert result == {"h7": 7, "h120": 120}
    assert all(len(chunk) <= deadtrees_seam.HASH_QUERY_CHUNK for chunk in calls)
    assert [h for chunk in calls for h in chunk] == hashes
