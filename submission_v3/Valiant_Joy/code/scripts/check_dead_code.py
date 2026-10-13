"""Dead-code audit using ast only - no linter is added to the pinned, offline-capable venv.

  python scripts/check_dead_code.py

Reports CANDIDATES to review, never a verdict: dynamic access (getattr, config-driven dispatch,
pytest collecting by name) makes any purely static answer wrong on its own. Two false positives are
expected and harmless - `from __future__ import annotations` in every module, and conftest's
`pytest_configure`, which pytest calls by name.

Four categories. The third is the one that earns its keep: a DEFAULT_*_CFG key nothing reads is a
knob that silently does nothing, which is worse than a missing one.
"""
import ast
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = ["core", "adapters", "scripts", "tests"]


def py_files():
    for pkg in PKGS:
        yield from sorted((ROOT / pkg).glob("*.py"))


def all_text():
    """Every file that could reference a name: python, yaml, md."""
    out = {}
    for p in list(py_files()) + sorted((ROOT / "configs").glob("*.yaml")) + [ROOT / "README.md"]:
        try:
            out[p] = p.read_text(encoding="utf-8")
        except Exception:
            pass
    return out


def main():
    texts = all_text()
    trees = {p: ast.parse(texts[p]) for p in py_files() if p in texts}

    # ---------------------------------------------------------------- unused imports
    print("=" * 78)
    print("UNUSED IMPORTS")
    print("=" * 78)
    n_imp = 0
    for p, tree in trees.items():
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported[(a.asname or a.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                # `from __future__ import annotations` is never referenced by name - it is a
                # compiler directive, and it is required for the postponed type hints this codebase
                # relies on. Reporting it produced 25 identical hits that buried every real finding,
                # so the check was worth less than the noise it generated.
                if node.module == "__future__":
                    continue
                for a in node.names:
                    if a.name != "*":
                        imported[a.asname or a.name] = node.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        # attribute roots: `np.array` -> np
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                base = n
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    used.add(base.id)
        src = texts[p]
        for name, line in sorted(imported.items(), key=lambda kv: kv[1]):
            if name in used:
                continue
            if re.search(rf'["\']{re.escape(name)}["\']', src) or "noqa" in src.splitlines()[line - 1]:
                continue
            print(f"  {p.relative_to(ROOT)}:{line}: {name}")
            n_imp += 1
    print(f"  -> {n_imp} candidate(s)\n")

    # ---------------------------------------------------------------- unreferenced definitions
    print("=" * 78)
    print("MODULE-LEVEL DEFS WITH NO REFERENCE ANYWHERE (py + yaml + README)")
    print("=" * 78)
    defs = defaultdict(list)      # name -> [(file, line, kind)]
    for p, tree in trees.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[node.name].append((p, node.lineno, type(node).__name__))
    n_def = 0
    for name, places in sorted(defs.items()):
        if name.startswith("test_") or name in ("main", "conftest"):
            continue
        hits = 0
        for p, text in texts.items():
            for m in re.finditer(rf"\b{re.escape(name)}\b", text):
                # skip the definition site itself
                line = text[: m.start()].count("\n") + 1
                if any(p == dp and line == dl for dp, dl, _ in places):
                    continue
                hits += 1
        if hits == 0:
            for p, line, kind in places:
                print(f"  {p.relative_to(ROOT)}:{line}: {kind[:-3].lower()} {name}()")
                n_def += 1
    print(f"  -> {n_def} candidate(s)\n")

    # ---------------------------------------------------------------- config keys never read
    print("=" * 78)
    print("DEFAULT_*_CFG / DEFAULTS KEYS NEVER READ IN CODE")
    print("=" * 78)
    allpy = "\n".join(texts[p] for p in trees)
    n_key = 0
    for p, tree in trees.items():
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            tgt = node.targets[0]
            nm = tgt.id if isinstance(tgt, ast.Name) else None
            if not nm or not (nm.startswith("DEFAULT") or nm == "DEFAULTS"):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for k in node.value.keys:
                if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                    continue
                key = k.value
                pat = rf'(\[\s*["\']{re.escape(key)}["\']\s*\]|\.get\(\s*["\']{re.escape(key)}["\']|'
                pat += rf'\bcfg\.{re.escape(key)}\b)'
                reads = len(re.findall(pat, allpy))
                yaml_use = any(re.search(rf"^\s*{re.escape(key)}\s*:", t, re.M)
                               for pp, t in texts.items() if pp.suffix == ".yaml")
                if reads == 0:
                    print(f"  {p.relative_to(ROOT)}: {nm}[{key!r}] never read"
                          f"{'  (but set in a yaml config!)' if yaml_use else ''}")
                    n_key += 1
    print(f"  -> {n_key} candidate(s)\n")

    # ---------------------------------------------------------------- files never referenced
    print("=" * 78)
    print("MODULES NEVER IMPORTED OR INVOKED")
    print("=" * 78)
    n_mod = 0
    for p in py_files():
        stem = p.stem
        if stem in ("__init__", "conftest") or stem.startswith("test_"):
            continue
        refs = 0
        for q, text in texts.items():
            if q == p:
                continue
            refs += len(re.findall(rf"\b{re.escape(stem)}\b", text))
        if refs == 0:
            print(f"  {p.relative_to(ROOT)}: never mentioned elsewhere")
            n_mod += 1
    print(f"  -> {n_mod} candidate(s)")


if __name__ == "__main__":
    main()
