#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import subprocess


# ==========================
# 工具
# ==========================
def esc(s):
    return (s or "").replace("\\", "\\\\").replace('"', "'")


def load_functions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("functions", [])


def find_function(funcs, name):
    for f in funcs:
        if f.get("function") == name:
            return f
    return None


# ==========================
# 產生 DOT
# ==========================
def write_dot(func_obj, out_base):

    ops = func_obj.get("operations", [])
    edges = func_obj.get("cfg_edges", [])

    op_map = {op["id"]: op for op in ops if "id" in op}

    lines = []
    lines.append("digraph CFG {")
    lines.append("  rankdir=TB;")
    lines.append('  graph [dpi=200, splines=ortho];')
    lines.append('  node [fontname="Consolas", fontsize=14, style="filled,rounded"];')
    lines.append('  edge [fontname="Consolas", fontsize=12];')
    lines.append("")

    # -------- nodes --------
    for oid, op in op_map.items():
        op_type = op.get("type", "")
        txt = op.get("text", "")

        if op_type == "join":
            label_text = ""  # JOIN 不顯示文字
        else:
            label_text = txt

        label = esc(label_text)

        lines.append(
            '"{}" [shape=box, label="{}"];'.format(oid, label)
        )

    lines.append("")

    # -------- edges --------
    for e in edges:
        a = e.get("from")
        b = e.get("to")
        if a in op_map and b in op_map:
            lab = esc(e.get("label", ""))
            if lab:
                lines.append('"{}" -> "{}" [label="{}"];'.format(a, b, lab))
            else:
                lines.append('"{}" -> "{}";'.format(a, b))

    lines.append("}")

    dot_path = str(out_base) + ".dot"
    Path(dot_path).write_text("\n".join(lines), encoding="utf-8")

    return dot_path


# ==========================
# 產生 SVG
# ==========================
def render_svg(dot_path, out_svg):
    subprocess.run(["dot", "-Tsvg", dot_path, "-o", out_svg], check=False)


# ==========================
# 主程式
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="output/functions.json")
    ap.add_argument("--func", required=True)
    args = ap.parse_args()

    funcs = load_functions(args.json)
    f = find_function(funcs, args.func)

    if not f:
        print("❌ Function not found")
        return

    # 🔥 固定輸出到 output/cfg/
    out_dir = Path("output") / "cfg"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_base = out_dir / (args.func + "_cfg")

    dot_path = write_dot(f, out_base)
    print("✅ DOT:", dot_path)

    svg_path = str(out_base) + ".svg"
    render_svg(dot_path, svg_path)
    print("✅ SVG:", svg_path)
    print("👉 用瀏覽器開 SVG，可無限放大")


if __name__ == "__main__":
    main()