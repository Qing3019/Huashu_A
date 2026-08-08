from __future__ import annotations

import hashlib
import json
from pathlib import Path


def find_q2_notebook() -> tuple[Path, dict]:
    candidates = []
    for path in Path(".").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if len(notebook.get("cells", [])) == 18:
            candidates.append((path, notebook))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one 18-cell Q2 notebook, found {len(candidates)}")
    return candidates[0]


q2_path, q2_notebook = find_q2_notebook()
core_indices = (2, 4, 6, 8)
core_parts = ["".join(q2_notebook["cells"][i]["source"]).rstrip() for i in core_indices]
addon = Path("_q3_first_passage_addon.py").read_text(encoding="utf-8").rstrip()
engine_text = "\n\n".join((*core_parts, addon)) + "\n"
engine_path = Path("_q3_engine_full.py")
engine_path.write_text(engine_text, encoding="utf-8")
# 使用正式生产文件的实际字节作为缓存签名，避免Windows换行转换造成歧义。
digest = hashlib.sha256(engine_path.read_bytes()).hexdigest()
Path("_q3_model_sha256.txt").write_text(digest + "\n", encoding="ascii")
print(f"Q2 source: {q2_path}")
print(f"wrote {engine_path} ({len(engine_text)} chars)")
print(f"model SHA-256: {digest}")
