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


class _TableQuery:
    """Chainable fake for one table: records filters, returns rows matching them."""

    def __init__(self, rows, filters):
        self.rows, self.filters, self.applied = rows, filters, []

    def select(self, _columns):
        return self

    def eq(self, column, value):
        self.applied.append(("eq", column, value))
        return self

    def gte(self, column, value):
        self.applied.append(("gte", column, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        self.filters.extend(self.applied)
        rows = [r for r in self.rows if all(r.get(c) == v for op, c, v in self.applied if op == "eq")]
        return SimpleNamespace(data=rows)


def _fake_platform(monkeypatch, tables):
    filters = []

    class FakeCommands:
        def _ensure_auth(self):
            return "token"

    @contextmanager
    def use_client(_token):
        yield SimpleNamespace(table=lambda name: _TableQuery(tables.get(name, []), filters))

    settings = SimpleNamespace(datasets_table="datasets", queue_table="queue", statuses_table="statuses")
    monkeypatch.setattr(deadtrees_seam, "_platform_api", lambda: (FakeCommands, use_client, settings))
    return filters


def test_find_dataset_reports_queued_processing(monkeypatch):
    filters = _fake_platform(monkeypatch, {
        "datasets": [{"id": 14269, "file_name": "a.tif"}],
        "queue": [{"id": 1, "dataset_id": 14269}],
        "statuses": [{"dataset_id": 14269, "current_status": "idle", "is_ortho_done": False, "has_error": False}],
    })
    found = deadtrees_seam.find_dataset("a.tif", created_after="2026-09-28T08:00:00+00:00")
    assert found == {"id": 14269, "processing_queued": True}
    # Only datasets created during this upload attempt count.
    assert ("gte", "created_at", "2026-09-28T08:00:00+00:00") in filters


def test_find_dataset_detects_processing_never_queued(monkeypatch):
    _fake_platform(monkeypatch, {
        "datasets": [{"id": 7, "file_name": "a.tif"}],
        "statuses": [{"dataset_id": 7, "current_status": "idle", "is_ortho_done": False, "has_error": False}],
    })
    assert deadtrees_seam.find_dataset("a.tif", "2026-09-28") == {"id": 7, "processing_queued": False}


def test_find_dataset_counts_finished_processing_as_queued(monkeypatch):
    _fake_platform(monkeypatch, {
        "datasets": [{"id": 7, "file_name": "a.tif"}],
        "statuses": [{"dataset_id": 7, "current_status": "idle", "is_ortho_done": True, "has_error": False}],
    })
    assert deadtrees_seam.find_dataset("a.tif", "2026-09-28")["processing_queued"] is True


def test_find_dataset_returns_none_when_the_file_did_not_land(monkeypatch):
    _fake_platform(monkeypatch, {"datasets": [{"id": 7, "file_name": "other.tif"}]})
    assert deadtrees_seam.find_dataset("a.tif", "2026-09-28") is None
