"""Renumber a manuscript's references in order of first citation.

The rewrite moved sections around, so the numbers inherited from the old paper no longer follow
the order a reader meets them. This maps every [n] in the body to its first-citation rank, rewrites
the body and the reference list together, and refuses to write if a cited number has no reference
or a reference is never cited.

    python scripts/paper/renumber_refs.py paper/v2/manuscript.md
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
body, refs = text.split("## References", 1)

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
order: list[int] = []
for m in CITE.finditer(body):
    for n in (int(x) for x in m.group(1).split(",")):
        if n not in order:
            order.append(n)

entries = {int(m.group(1)): m.group(2).strip()
           for m in re.finditer(r"^\[(\d+)\]\s*(.+)$", refs, flags=re.M)}
missing = [n for n in order if n not in entries]
uncited = [n for n in entries if n not in order]
if missing or uncited:
    raise SystemExit(f"refusing: cited without entry {missing}, entry never cited {uncited}")

new = {old: i + 1 for i, old in enumerate(order)}


def renum(m: re.Match) -> str:
    nums = sorted(new[int(x)] for x in m.group(1).split(","))
    return "[" + ", ".join(str(n) for n in nums) + "]"


body2 = CITE.sub(renum, body)
refs2 = "\n\n".join(f"[{new[old]}] {entries[old]}" for old in sorted(entries, key=lambda o: new[o]))
path.write_text(body2 + "## References\n\n" + refs2 + "\n", encoding="utf-8")
moved = sum(1 for o, n in new.items() if o != n)
print(f"{len(new)} references renumbered by first citation ({moved} changed number)")
