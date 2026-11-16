"""Offline CAD convex decomposition. Runtime only needs the generated assets.

Install requirements-collision.txt, then run:
    python -m integrations.build_collision_meshes
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from sourccey.build import MODEL, ROOT, SOURCE, vertices


OUTPUT = ROOT / "models/collision"
PARAMETERS = dict(threshold=.001, real_metric=True, preprocess_mode="auto",
                  resolution=1000, mcts_nodes=10, mcts_iterations=10,
                  mcts_max_depth=2, seed=7)


def decompose(job):
    import coacd
    coacd.set_log_level("error")
    name, filename, scale, output = job
    path = SOURCE.parent / filename
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256((source_hash + json.dumps(scale) + json.dumps(PARAMETERS, sort_keys=True)).encode()).hexdigest()[:20]
    target = Path(output) / (key + ".npz")
    meta = target.with_suffix(".json")
    if target.exists() and meta.exists():
        result = json.loads(meta.read_text())
        result["source"] = filename
        return name, result
    started = time.perf_counter()
    points = vertices(path) * scale
    verts, indices = np.unique(points, axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    # Skip expensive remeshing for a closed two-manifold CAD solid.
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    parameters = PARAMETERS.copy()
    if np.all(counts == 2):
        parameters["preprocess_mode"] = "off"
    parts = coacd.run_coacd(coacd.Mesh(verts, faces), **parameters)
    arrays = {}
    for i, (v, f) in enumerate(parts):
        arrays[f"v{i}"] = v
        arrays[f"f{i}"] = f
    np.savez_compressed(target, **arrays)
    result = {"asset": target.name, "source": filename, "source_sha256": source_hash,
              "scale": scale, "parts": len(parts), "parameters": parameters,
              "asset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "build_wall_s": round(time.perf_counter() - started, 2)}
    meta.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return name, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(MODEL))
    names = sorted({g.meshname for g in spec.geoms if g.name.startswith("collision_")})
    # Decompose shared source bytes once, including when aliases are queued at
    # the same time. The mirrored assembly contains many identical STL copies.
    unique, aliases = {}, {}
    for name in names:
        mesh = spec.mesh(name)
        key = (hashlib.sha256((SOURCE.parent / mesh.file).read_bytes()).hexdigest(), tuple(mesh.scale))
        if key not in unique:
            unique[key] = (name, mesh.file, list(mesh.scale), str(args.output))
        aliases.setdefault(unique[key][0], []).append(name)
    jobs = list(unique.values())
    # Keep OpenMP from oversubscribing every CPU in every independent worker.
    os.environ["OMP_NUM_THREADS"] = "2"
    from importlib.metadata import version
    manifest = {"generator": "CoACD", "version": version("coacd"), "parameters": PARAMETERS,
                "units": "meters; vertices preserve source mesh coordinates", "meshes": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(decompose, job) for job in jobs]):
            name, result = future.result()
            for alias in aliases[name]:
                manifest["meshes"][alias] = {**result, "source": spec.mesh(alias).file}
            print(f"{len(manifest['meshes'])}/{len(names)} mesh references: {name}: {result['parts']} convex parts ({result['build_wall_s']} s)", flush=True)
            (args.output / "manifest.partial.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["meshes"] = dict(sorted(manifest["meshes"].items()))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "manifest.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
