import hashlib
import json
import subprocess

import pytest

from scripts.linux_handoff import ROOT, digest, restore_inputs, verify_dependency


def test_clone_restores_original_mapping_bytes_and_preserves_existing_calibration(tmp_path):
    bundle = tmp_path / 'handoff/linux'
    bundle.mkdir(parents=True)
    raw = b'{"reference": true}\r\n'
    (bundle / 'robot_joint_mapping.json').write_bytes(raw)
    (bundle / 'manifest.json').write_text(json.dumps(
        {'joint_mapping_sha256': hashlib.sha256(raw).hexdigest()}))
    restore_inputs(tmp_path)
    target = tmp_path / 'artifacts/robot_joint_mapping.json'
    assert target.read_bytes() == raw
    restore_inputs(tmp_path)
    target.write_bytes(raw.replace(b'\r\n', b'\n'))
    restore_inputs(tmp_path)
    assert target.read_bytes() == raw
    assert next(target.parent.glob('robot_joint_mapping.before_handoff_*.json')).read_bytes() == raw.replace(b'\r\n', b'\n')
    target.write_bytes(b'other calibration')
    with pytest.raises(ValueError, match='preserved without overwriting'):
        restore_inputs(tmp_path)
    assert target.read_bytes() == b'other calibration'


def test_versioned_mapping_and_dependency_inventory_are_complete():
    bundle = ROOT / 'handoff/linux'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert digest(bundle / 'robot_joint_mapping.json') == manifest['joint_mapping_sha256']
    assert {d['name'] for d in manifest['dependencies']} == {'lerobot-vulcan', 'lerobot-robot-sourccey'}
    for spec in manifest['dependencies']:
        assert len(spec['revision']) == 40
        assert {'pyproject.toml', 'uv.lock'} <= spec['source_sha256'].keys()


def test_dependency_verification_accepts_newlines_but_rejects_modified_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: 'pinned\n')
    path = tmp_path / 'source.py'
    path.write_bytes(b'original\r\n')
    spec = dict(revision='pinned', source_sha256={'source.py': hashlib.sha256(b'original\n').hexdigest()})
    verify_dependency(tmp_path, spec)
    path.write_bytes(b'modified\n')
    with pytest.raises(ValueError, match='source differs'):
        verify_dependency(tmp_path, spec)
