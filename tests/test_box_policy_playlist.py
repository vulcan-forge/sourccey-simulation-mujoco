import json

import pytest

from examples.box_policy_playlist import attempts


def saved_sweep(tmp_path, count=8):
    rows = []
    for index in range(count):
        name = f"scene_{index}"
        folder = tmp_path/name
        folder.mkdir()
        for filename in ("scene.json", "recording.npz", "report.json"):
            (folder/filename).write_text("placeholder")
        rows.append(dict(episode=name))
    (tmp_path/"summary.json").write_text(json.dumps(dict(format="sourccey-xvla-sweep-v1", results=rows)))
    return rows


def test_shuffle_has_no_duplicates_and_seed_reproduces_selection(tmp_path):
    rows = saved_sweep(tmp_path)
    shuffled = attempts(tmp_path, seed=7)
    assert {p.name for p in shuffled} == {r["episode"] for r in rows}
    assert len(shuffled) == len(set(shuffled)) == len(rows)
    assert shuffled == attempts(tmp_path, seed=7)
    assert shuffled != attempts(tmp_path, seed=9)
    assert attempts(tmp_path, seed=7, count=3) == shuffled[:3]


def test_playlist_rejects_missing_recordings_and_paths_outside_sweep(tmp_path):
    saved_sweep(tmp_path, count=1)
    (tmp_path/"scene_0"/"recording.npz").unlink()
    with pytest.raises(ValueError, match="Missing"):
        attempts(tmp_path)
    (tmp_path/"summary.json").write_text(json.dumps(dict(format="sourccey-xvla-sweep-v1",
                                                        results=[dict(episode="../outside")])))
    with pytest.raises(ValueError, match="Invalid"):
        attempts(tmp_path)
