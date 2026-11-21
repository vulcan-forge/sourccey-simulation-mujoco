"""Restore versioned reference inputs and pinned source dependencies, without data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'handoff/linux'


def digest(path, *, source=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if source else data).hexdigest()


def restore_inputs(root=ROOT):
    root = Path(root)
    bundle = root / 'handoff/linux'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    source = bundle / 'robot_joint_mapping.json'
    expected = manifest['joint_mapping_sha256']
    if digest(source) != expected:
        raise ValueError('Versioned joint mapping checksum does not match the handoff')
    target = root / 'artifacts/robot_joint_mapping.json'
    if target.exists():
        if digest(target) != expected:
            if target.read_bytes().replace(b'\r\n', b'\n') != source.read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError(f'Existing mapping differs; preserved without overwriting: {target}')
            backup = target.with_name(f'robot_joint_mapping.before_handoff_{digest(target)[:12]}.json')
            if not backup.exists():
                backup.write_bytes(target.read_bytes())
            target.write_bytes(source.read_bytes())
            print(f'Restored reference newline bytes; previous copy preserved: {backup}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep the original bytes, including newlines, for metadata fingerprints.
        target.write_bytes(source.read_bytes())
    print(f'Verified reference input: {target}')


def verify_dependency(target, spec):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
    if revision != spec['revision']:
        raise ValueError(f'Wrong dependency revision at {target}: {revision}')
    for name, expected in spec['source_sha256'].items():
        path = target / name
        if not path.is_file() or digest(path, source=True) != expected:
            raise ValueError(f'Dependency source differs from the handoff: {path}')


def dependencies(destination):
    manifest = json.loads((HANDOFF / 'manifest.json').read_text(encoding='utf-8'))
    destination = Path(destination).resolve()
    for spec in manifest['dependencies']:
        target = destination / spec['directory']
        patch = HANDOFF / spec['patch'] if spec.get('patch') else None
        if patch and digest(patch) != spec['patch_sha256']:
            raise ValueError(f'Dependency patch checksum changed: {patch}')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Avoid downloading unrelated Git LFS test artifacts and weights.
            subprocess.run(['git', '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false',
                            'clone', '--no-checkout', spec['url'], str(target)], check=True)
            subprocess.run(['git', '-c', 'filter.lfs.process=', '-c', 'filter.lfs.required=false',
                            'checkout', '--detach', spec['revision']], cwd=target, check=True)
            if patch and patch.stat().st_size:
                subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, check=True)
                subprocess.run(['git', 'apply', str(patch)], cwd=target, check=True)
        # Existing directories are checked, never reset or silently patched.
        verify_dependency(target, spec)
        print(f'Verified pinned dependency: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('restore-inputs')
    deps = sub.add_parser('dependencies')
    deps.add_argument('--destination', type=Path, default=ROOT / 'artifacts/linux_deps')
    args = parser.parse_args()
    if args.command == 'restore-inputs':
        restore_inputs()
    else:
        dependencies(args.destination)


if __name__ == '__main__':
    main()
