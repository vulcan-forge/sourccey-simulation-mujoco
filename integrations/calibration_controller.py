"""Run the robot's unchanged calibration decisions on a cooperative sim clock.

Only blocking sleeps and calls to methods containing sleeps become generators.
Step sizes, conditions, interpolation, retries and returned values stay intact.
The vendored controllers never import a serial/GPIO driver.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict


REFERENCE = Path(__file__).with_name("true_calibration_reference")


def controller(name, clock, cooperative=True, logger=None):
    path = REFERENCE / name
    manifest = json.loads((REFERENCE / "provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["reference_sha256"][name]:
        raise ValueError(f"Calibration reference checksum mismatch: {name}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # time is supplied by the scene. No hardware modules exist in these extracts.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(n.name == "time" for n in node.names))]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for cls in classes:
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        sleeping = {name for name, node in methods.items() if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "time"
            and n.func.attr == "sleep" for n in ast.walk(node))}
        while True:
            indirect = {name for name, node in methods.items() if any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"
                and n.func.attr in sleeping for n in ast.walk(node))}
            if indirect <= sleeping:
                break
            sleeping |= indirect

        class Cooperate(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    owner, method = node.func.value.id, node.func.attr
                    if owner == "time" and method == "sleep":
                        return ast.copy_location(ast.Yield(node.args[0]), node)
                    if owner == "self" and method in sleeping:
                        return ast.copy_location(ast.YieldFrom(node), node)
                return node

        if cooperative:
            Cooperate().visit(cls)
    namespace = {"time": clock, "logger": logger or logging.getLogger("true_calibration.controller"),
                 "Dict": Dict, "__name__": __name__}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace
