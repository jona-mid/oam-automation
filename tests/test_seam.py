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


def test_file_name_lookup_is_chunked_with_one_login(monkeypatch):
    calls, logins = [], []

    class FakeQuery:
        def select(self, _columns):
            return self

        def in_(self, _column, values):
            self.values = list(values)
            return self

        def execute(self):
            calls.append(self.values)
            return SimpleNamespace(data=[{"file_name": n.upper()} for n in self.values if n in {"f7.tif", "f99.tif"}])

    class FakeCommands:
        def _ensure_auth(self):
            logins.append(1)
            return "token"

    @contextmanager
    def use_client(_token):
        yield SimpleNamespace(table=lambda _name: FakeQuery())

    monkeypatch.setattr(
        deadtrees_seam,
        "_platform_api",
        lambda: (FakeCommands, use_client, SimpleNamespace(datasets_table="v2_datasets")),
    )
    names = [f"f{i}.tif" for i in range(120)]
    assert deadtrees_seam.file_names_on_platform(names) == {"f7.tif", "f99.tif"}
    assert len(logins) == 1
    assert all(len(chunk) <= deadtrees_seam.HASH_QUERY_CHUNK for chunk in calls)
    assert [n for chunk in calls for n in chunk] == names
