import json
from pathlib import Path
import re
from collections import defaultdict

INPUT_JSON = "output/led_cases.json"
OUT_DIR = Path("output/flow_v2")

# ---- styling ----
def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def is_led_write_sink(fn: str) -> bool:
    return "Wr_AllArea_BGRData" in fn

def is_brightness_sink(fn: str) -> bool:
    return "Brightness_Transfer" in fn

def is_formula_sink(fn: str) -> bool:
    return "formula" in fn or "HsvToRgb" in fn

def node_style(fn: str) -> str:
    if is_led_write_sink(fn):
        return '[shape=box, style=filled, fillcolor=lightcoral]'
    if is_brightness_sink(fn):
        return '[shape=box, style=filled, fillcolor=orange]'
    if is_formula_sink(fn):
        return '[shape=box, style=filled, fillcolor=plum]'
    if fn.endswith("_Service") or fn.endswith("_Init"):
        return '[shape=box, style=filled, fillcolor=lightgray]'
    return '[shape=box]'

# ---- cond utilities ----
SWITCH_RE = re.compile(r"switch\((.*?)\)")
CASE_RE = re.compile(r"case\((.*?)\)")

def cond_kind(c: str) -> str:
    c = c.strip()
    if c.startswith("switch("):
        return "switch"
    if c.startswith("case("):
        return "case"
    if c.startswith("if("):
        return "if"
    if c.startswith("else_if("):
        return "else_if"
    if c.startswith("else("):
        return "else"
    if c.startswith("pp("):
        return "pp"
    if c.startswith("for(") or c.startswith("while(") or c.startswith("do_while("):
        return "loop"
    if c.startswith("ternary_"):
        return "ternary"
    return "cond"

def normalize_conds(conds):
    # drop ultra-noisy duplicates while keeping order
    seen = set()
    out = []
    for c in conds or []:
        c = (c or "").strip()
        if not c:
            continue
        key = c
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def chain_from_edgeconds(edge_conds_list):
    """
    edge_conds_list: list per-edge conds (root->n1, n1->n2, ...)
    returns a flattened decision chain in order, normalized.
    """
    chain = []
    for edge_conds in edge_conds_list or []:
        chain.extend(edge_conds or [])
    return normalize_conds(chain)

# ---- graph building ----
def build_case_flow(case_name, case_data, root):
    """
    Build a decision-chain merged flow graph:
    - Choose a few representative paths (dedupe by (chain, sink, last_service))
    - Each path: root -> decision nodes (chain) -> services -> sink
    """
    paths = case_data.get("paths", [])
    sinks = case_data.get("sinks", [])
    entries = case_data.get("entries", [])

    # Deduplicate representative paths
    reps = []
    seen = set()

    for p in paths:
        nodes = p.get("nodes", [])
        edge_conds = p.get("edge_conds", [])
        sink = p.get("sink")

        chain = chain_from_edgeconds(edge_conds)
        # last "real" function on path
        last_fn = None
        for n in reversed(nodes):
            if n and not n.startswith("…") and not n.startswith("↻"):
                last_fn = n
                break

        key = (" && ".join(chain), sink or "", last_fn or "")
        if key in seen:
            continue
        seen.add(key)
        reps.append({
            "nodes": nodes,
            "sink": sink,
            "chain": chain,
        })

        # limit to keep graph readable (tune if needed)
        if len(reps) >= 25:
            break

    return {
        "case": case_name,
        "entries": entries,
        "sinks": sinks,
        "representative_paths": reps
    }

def emit_dot_flow(case_flow, root):
    """
    DOT output (flow):
    root -> decision chain -> function nodes -> sink nodes
    """
    case_name = case_flow["case"]
    reps = case_flow["representative_paths"]

    dot = []
    dot.append("digraph G {")
    dot.append("rankdir=LR;")
    dot.append('node [fontname="Consolas"];')
    dot.append('edge [fontname="Consolas"];')

    dot.append(f'"{root}" [shape=oval, style=filled, fillcolor=lightblue];')

    # A "CASE" header node for clarity
    case_hdr = f"CASE::{case_name}"
    dot.append(f'"{case_hdr}" [shape=diamond, style=filled, fillcolor=lightyellow, label="case({case_name})"];')
    dot.append(f'"{root}" -> "{case_hdr}";')

    # Create nodes/styles on demand
    made = set()

    def ensure_node(n):
        if n in made:
            return
        made.add(n)
        # decision nodes
        if n.startswith("DEC::"):
            label = n[len("DEC::"):]
            dot.append(f'"{n}" [shape=diamond, style=filled, fillcolor=lightyellow, label="{label}"];')
        else:
            dot.append(f'"{n}" {node_style(n)};')

    edge_seen = set()

    def add_edge(a, b):
        k = (a, b)
        if k in edge_seen:
            return
        edge_seen.add(k)
        dot.append(f'"{a}" -> "{b}";')

    for rep in reps:
        chain = rep["chain"]
        nodes = rep["nodes"]
        sink = rep.get("sink")

        # decision chain nodes
        prev = case_hdr
        for c in chain:
            dn = "DEC::" + c
            ensure_node(dn)
            add_edge(prev, dn)
            prev = dn

        # now connect to the functional path (skip root, skip max_depth/cycle markers)
        fn_path = [n for n in nodes if n and n != root and not n.startswith("…") and not n.startswith("↻")]
        # make it shorter: keep only key functions (Init/Service/Update + sinks)
        key_fn = []
        for n in fn_path:
            if n.endswith("_Init") or n.endswith("_Service") or n.startswith("Update_") or is_led_write_sink(n) or is_brightness_sink(n) or is_formula_sink(n):
                key_fn.append(n)

        # fallback if empty
        if not key_fn and fn_path:
            key_fn = fn_path[-3:]

        for fn in key_fn:
            ensure_node(fn)
        for i, fn in enumerate(key_fn):
            if i == 0:
                add_edge(prev, fn)
            else:
                add_edge(key_fn[i-1], fn)

        # ensure sink highlighted even if not in key list
        if sink and sink not in key_fn:
            ensure_node(sink)
            if key_fn:
                add_edge(key_fn[-1], sink)
            else:
                add_edge(prev, sink)

    dot.append("}")
    return "\n".join(dot)

def emit_dot_arch(case_flow, root):
    """
    Architecture doc view:
    root -> key services/updates -> sinks (no decision nodes)
    """
    case_name = case_flow["case"]
    reps = case_flow["representative_paths"]

    dot = []
    dot.append("digraph G {")
    dot.append("rankdir=LR;")
    dot.append('node [fontname="Consolas"];')
    dot.append('edge [fontname="Consolas"];')

    dot.append(f'"{root}" [shape=oval, style=filled, fillcolor=lightblue];')

    case_hdr = f"CASE::{case_name}"
    dot.append(f'"{case_hdr}" [shape=box, style=filled, fillcolor=white, label="{case_name}"];')
    dot.append(f'"{root}" -> "{case_hdr}";')

    made = set()
    def ensure_node(n):
        if n in made:
            return
        made.add(n)
        dot.append(f'"{n}" {node_style(n)};')

    edge_seen = set()
    def add_edge(a, b):
        k = (a, b)
        if k in edge_seen:
            return
        edge_seen.add(k)
        dot.append(f'"{a}" -> "{b}";')

    # aggregate
    services = set()
    updates = set()
    sinks = set()

    for rep in reps:
        for n in rep["nodes"]:
            if not n or n == root or n.startswith("…") or n.startswith("↻"):
                continue
            if n.endswith("_Service") or n.endswith("_Init"):
                services.add(n)
            if n.startswith("Update_"):
                updates.add(n)
            if is_led_write_sink(n) or is_brightness_sink(n) or is_formula_sink(n):
                sinks.add(n)
        if rep.get("sink"):
            sinks.add(rep["sink"])

    for n in sorted(services | updates | sinks):
        ensure_node(n)

    # Connect: case -> services/updates; services/updates -> sinks
    for s in sorted(services | updates):
        add_edge(case_hdr, s)
    for s in sorted(services | updates):
        for k in sorted(sinks):
            # only connect plausible links to keep readability
            if (s.endswith("_Service") and (is_led_write_sink(k) or is_formula_sink(k))) or (s.startswith("Update_") and is_brightness_sink(k)):
                add_edge(s, k)

    dot.append("}")
    return "\n".join(dot)

def emit_llm_lite(case_flow):
    """
    Token-efficient JSON for LLM:
    - case
    - representative decision chains
    - key sinks
    """
    out_paths = []
    for rep in case_flow["representative_paths"]:
        out_paths.append({
            "chain": rep["chain"],
            "key_nodes": [n for n in rep["nodes"] if n and not n.startswith("…") and not n.startswith("↻")][-8:],  # last few nodes only
            "sink": rep.get("sink")
        })

    return {
        "case": case_flow["case"],
        "entries": case_flow["entries"],
        "sinks": case_flow["sinks"],
        "paths": out_paths[:20],  # cap
    }

def main():
    data = json.loads(Path(INPUT_JSON).read_text(encoding="utf-8"))
    root = data["root"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dot_flow").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dot_arch").mkdir(parents=True, exist_ok=True)

    lite_all = {
        "root": root,
        "cases": {}
    }

    for case_name, case_data in data.get("cases", {}).items():
        cf = build_case_flow(case_name, case_data, root)

        dot_flow = emit_dot_flow(cf, root)
        dot_arch = emit_dot_arch(cf, root)
        lite = emit_llm_lite(cf)

        (OUT_DIR / "dot_flow" / f"{sanitize(case_name)}.dot").write_text(dot_flow, encoding="utf-8")
        (OUT_DIR / "dot_arch" / f"{sanitize(case_name)}.dot").write_text(dot_arch, encoding="utf-8")

        lite_all["cases"][case_name] = lite

    (OUT_DIR / "led_cases_lite.json").write_text(
        json.dumps(lite_all, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ Generated:")
    print(f"  - {OUT_DIR / 'dot_flow'}  (debug flow, decision-chain merged)")
    print(f"  - {OUT_DIR / 'dot_arch'}  (architecture doc view)")
    print(f"  - {OUT_DIR / 'led_cases_lite.json'}  (LLM-friendly JSON)")

if __name__ == "__main__":
    main()