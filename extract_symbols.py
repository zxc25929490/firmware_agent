import os
import re
import json
import argparse
from pathlib import Path

DEFINE_HEX_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(0x[0-9A-Fa-f]+)\b")

def iter_header_files(codebase: Path, prefer_names: list[str] | None = None):
    prefer = set(prefer_names or [])
    preferred = []
    others = []
    for p in codebase.rglob("*.h"):
        if p.name in prefer:
            preferred.append(p)
        else:
            others.append(p)
    # 先掃你指定的檔名（例如 memory.h），再掃其他 .h
    yield from preferred
    yield from others

def extract_defines_from_file(path: Path) -> dict:
    out = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for line in text:
        m = DEFINE_HEX_RE.match(line)
        if not m:
            continue
        name, hexv = m.group(1), m.group(2)
        # 後者覆蓋前者（有些專案會重複 define）
        out[name] = hexv.lower()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebase", default="codebase", help="source root folder")
    ap.add_argument("--out", default="output/symbols.json", help="output json path")
    ap.add_argument("--prefer", default="memory.h", help="preferred header name (comma-separated)")
    args = ap.parse_args()

    codebase = Path(args.codebase).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prefer_names = [x.strip() for x in args.prefer.split(",") if x.strip()]

    symbols = {}
    scanned = 0
    for hf in iter_header_files(codebase, prefer_names=prefer_names):
        scanned += 1
        defs = extract_defines_from_file(hf)
        if defs:
            symbols.update(defs)

    payload = {
        "codebase": str(codebase),
        "header_scanned": scanned,
        "symbol_count": len(symbols),
        "symbols": symbols,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] symbols.json written: {out_path}")
    print(f"     scanned headers: {scanned}, symbols: {len(symbols)}")

if __name__ == "__main__":
    main()