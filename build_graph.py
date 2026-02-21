import json
import argparse
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="output/symbols.json")
    ap.add_argument("--functions", default="output/functions.json")
    ap.add_argument("--out", default="output/functions_enriched.json")
    args = ap.parse_args()

    symbols_p = Path(args.symbols).resolve()
    funcs_p = Path(args.functions).resolve()
    out_p = Path(args.out).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    symbols_payload = load_json(symbols_p) if symbols_p.exists() else {"symbols": {}}
    symbols = symbols_payload.get("symbols", {}) or {}
    symbol_names = set(symbols.keys())

    funcs_payload = load_json(funcs_p)
    functions = funcs_payload.get("functions", [])

    # MVP: 用「字面命中」先做初步標記
    for f in functions:
        reg_used = set(f.get("reg_symbols_used", []) or [])
        for c in f.get("calls", []) or []:
            if c in symbol_names:
                reg_used.add(c)
        f["reg_symbols_used"] = sorted(reg_used)

    out_payload = {
        "codebase": funcs_payload.get("codebase"),
        "function_count": len(functions),
        "functions": functions,
        "note": "Phase1 MVP: reg_symbols_used uses simple literal hit; Phase2 will parse call args/assignments to resolve symbols properly."
    }

    out_p.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] enriched json written: {out_p}")
    print(f"     symbols: {len(symbols)}, functions: {len(functions)}")

if __name__ == "__main__":
    main()