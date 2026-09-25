"""Resume behavior of the VLM attempt log in aerial_phenology_audit."""

import json

import aerial_phenology_audit as audit


def _write_attempts(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_attempts_from_an_older_prompt_are_ignored(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    _write_attempts(
        attempts,
        [
            {"status": "success", "image_id": "old", "prompt_version": "phenology-v2"},
            {"status": "success", "image_id": "new", "prompt_version": audit.PROMPT_VERSION},
        ],
    )
    assert audit.successful_ids(attempts) == {"new"}
    assert set(audit.latest_by_image(attempts)) == {"new"}


def test_latest_attempt_with_the_current_prompt_wins(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    _write_attempts(
        attempts,
        [
            {"status": "error", "image_id": "a", "prompt_version": audit.PROMPT_VERSION},
            {"status": "success", "image_id": "a", "prompt_version": audit.PROMPT_VERSION},
            {"status": "error", "image_id": "a", "prompt_version": "phenology-v2"},
        ],
    )
    assert audit.latest_by_image(attempts)["a"]["status"] == "success"
