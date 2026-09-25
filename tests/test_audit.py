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


def test_extract_json_unwraps_a_single_element_list():
    content = '[{"tree_canopy_leaf_state": "leaf_on", "visual_cue": "dense green crowns throughout"}]'
    assert audit.extract_json(content)["tree_canopy_leaf_state"] == "leaf_on"


def test_extract_json_still_rejects_other_lists():
    import pytest

    with pytest.raises(ValueError):
        audit.extract_json('[{"a": 1}, {"b": 2}]')


def _jpeg(path, size, box=None):
    from PIL import Image

    image = Image.new("RGB", size, (0, 0, 0))
    if box:
        image.paste((40, 120, 40), box)
    image.save(path, "JPEG")
    return path


def test_mostly_black_preview_is_excluded(tmp_path):
    path = _jpeg(tmp_path / "a.jpeg", (1200, 1200), box=(0, 0, 200, 200))  # ~3% image
    assert audit.preview_info(path, 1000)[0] == "mostly_nodata"


def test_preview_with_enough_image_is_eligible(tmp_path):
    path = _jpeg(tmp_path / "b.jpeg", (1200, 1200), box=(0, 0, 600, 1200))  # 50% image
    assert audit.preview_info(path, 1000)[0] == "eligible"


def test_small_preview_is_still_too_small(tmp_path):
    path = _jpeg(tmp_path / "c.jpeg", (400, 400), box=(0, 0, 400, 400))
    assert audit.preview_info(path, 1000)[0] == "too_small"
