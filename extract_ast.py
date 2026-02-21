import os
import json
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_c

# ==========================
# Parser 初始化
# ==========================
parser = Parser()
C_LANGUAGE = Language(tree_sitter_c.language(), "c")
parser.set_language(C_LANGUAGE)


# ==========================
# 工具函式
# ==========================
def node_text(src: bytes, node) -> str:
    if node is None:
        return ""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def one_line(s: str, limit: int = 160) -> str:
    """Normalize to one line for condition labels."""
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) > limit:
        return s[:limit] + "..."
    return s


# ==========================
# 核心抽取
# ==========================
def extract_functions_from_file(path, codebase_root: Path):
    src = Path(path).read_bytes()

    tree = parser.parse(src)
    root = tree.root_node

    results = []

    for node in root.children:
        if node.type != "function_definition":
            continue

        func_name = ""
        decl = node.child_by_field_name("declarator")

        if decl:
            for sub in walk(decl):
                if sub.type == "identifier":
                    func_name = node_text(src, sub)
                    break

        if not func_name:
            continue

        body = node.child_by_field_name("body")

        # -------------------------------------------------
        # Calls + branch conditions + loops extraction
        # -------------------------------------------------
        callsites = []

        def add_call(call_node, conds):
            fn_node = call_node.child_by_field_name("function")
            if not fn_node:
                return
            callee = node_text(src, fn_node).strip()
            if not callee:
                return

            # tree-sitter point is 0-based (row, column)
            row, col = call_node.start_point

            callsites.append({
                "callee": callee,
                "conds": conds[:] if conds else [],
                "loc": {"row": row + 1, "col": col + 1},
            })

        def push(conds, label: str):
            label = one_line(label)
            if not label:
                return conds
            return conds + [label]

        def visit(stmt, conds):
            """
            Traverse with a running list of control-flow conditions.

            We attach conds for:
            - if / else-if / else
            - switch / case / default
            - for / while / do-while
            - ternary (?:)
            - preprocessor: #if / #ifdef / #ifndef / #elif / #else
            """
            if stmt is None:
                return

            t = stmt.type

            # -------- function call --------
            if t == "call_expression":
                add_call(stmt, conds)
                # still traverse args in case of nested calls
                for ch in stmt.children:
                    visit(ch, conds)
                return

            # -------- if / else --------
            if t == "if_statement":
                cond_node = stmt.child_by_field_name("condition")
                cond_txt = one_line(node_text(src, cond_node)) if cond_node else "<cond>"

                cons = stmt.child_by_field_name("consequence")
                alt = stmt.child_by_field_name("alternative")

                visit(cons, push(conds, f"if({cond_txt})"))
                if alt:
                    # tree-sitter puts "else if" as alternative being another if_statement
                    if alt.type == "if_statement":
                        visit(alt, push(conds, f"else_if({cond_txt})"))
                    else:
                        visit(alt, push(conds, f"else({cond_txt})"))
                return

            # -------- switch / case / default --------
            if t == "switch_statement":
                val_node = stmt.child_by_field_name("condition") or stmt.child_by_field_name("value")
                val_txt = one_line(node_text(src, val_node)) if val_node else "<expr>"
                body_node = stmt.child_by_field_name("body")
                if body_node:
                    for ch in body_node.children:
                        visit(ch, push(conds, f"switch({val_txt})"))
                return

            # case/default live under switch body
            if t in ("case_statement", "default_statement"):
                label_txt = "default"
                if t == "case_statement":
                    val = stmt.child_by_field_name("value")
                    label_txt = one_line(node_text(src, val)) if val else "<case>"

                # Visit children excluding label expression itself
                for ch in stmt.children:
                    if t == "case_statement" and ch == stmt.child_by_field_name("value"):
                        continue
                    visit(ch, push(conds, f"case({label_txt})"))
                return

            # -------- loops --------
            # for(...) { ... }
            if t == "for_statement":
                init_node = stmt.child_by_field_name("initializer")
                cond_node = stmt.child_by_field_name("condition")
                upd_node = stmt.child_by_field_name("update")
                body_node = stmt.child_by_field_name("body")

                init_txt = one_line(node_text(src, init_node)) if init_node else ""
                cond_txt = one_line(node_text(src, cond_node)) if cond_node else ""
                upd_txt = one_line(node_text(src, upd_node)) if upd_node else ""

                label = "for("
                label += init_txt if init_txt else ""
                label += "; "
                label += cond_txt if cond_txt else ""
                label += "; "
                label += upd_txt if upd_txt else ""
                label += ")"

                # Calls in init/cond/update are also relevant; attach same loop label
                if init_node:
                    visit(init_node, push(conds, label))
                if cond_node:
                    visit(cond_node, push(conds, label))
                if upd_node:
                    visit(upd_node, push(conds, label))
                if body_node:
                    visit(body_node, push(conds, label))
                return

            # while(cond) { ... }
            if t == "while_statement":
                cond_node = stmt.child_by_field_name("condition")
                body_node = stmt.child_by_field_name("body")
                cond_txt = one_line(node_text(src, cond_node)) if cond_node else "<cond>"
                label = f"while({cond_txt})"
                if cond_node:
                    visit(cond_node, push(conds, label))
                if body_node:
                    visit(body_node, push(conds, label))
                return

            # do { ... } while(cond);
            if t == "do_statement":
                cond_node = stmt.child_by_field_name("condition")
                body_node = stmt.child_by_field_name("body")
                cond_txt = one_line(node_text(src, cond_node)) if cond_node else "<cond>"
                label = f"do_while({cond_txt})"
                if body_node:
                    visit(body_node, push(conds, label))
                if cond_node:
                    visit(cond_node, push(conds, label))
                return

            # -------- ternary operator (cond ? a : b) --------
            if t == "conditional_expression":
                c = stmt.child_by_field_name("condition")
                cons = stmt.child_by_field_name("consequence")
                alt = stmt.child_by_field_name("alternative")
                cond_txt = one_line(node_text(src, c)) if c else "<cond>"

                visit(cons, push(conds, f"ternary_true({cond_txt})"))
                visit(alt, push(conds, f"ternary_false({cond_txt})"))

                # also walk condition (it may contain calls)
                if c:
                    visit(c, push(conds, f"ternary_cond({cond_txt})"))
                return

            # -------- preprocessor conditionals --------
            # Depending on grammar version, types can include:
            # preproc_if / preproc_ifdef / preproc_ifndef / preproc_elif / preproc_else / preproc_endif
            if t.startswith("preproc_"):
                directive = one_line(node_text(src, stmt))
                # Keep only first directive line as label (avoid swallowing big blocks)
                first_line = directive.split(" ", 1)[0] if directive else t
                # Better label for common directives:
                label = directive if directive.startswith("#") else f"#{t}"
                # Traverse children with preprocessor label
                for ch in stmt.children:
                    visit(ch, push(conds, f"pp({label})"))
                return

            # -------- generic recursion --------
            for ch in stmt.children:
                visit(ch, conds)

        # Run visitor on function body
        if body:
            visit(body, [])

        calls = sorted(set([c["callee"] for c in callsites]))

        results.append({
            "function": func_name,
            "file": str(Path(path).relative_to(codebase_root)).replace("\\", "/"),
            "calls": calls,
            "callsites": callsites,
        })

    return results


# ==========================
# 建 graph + reverse map
# ==========================
def build_graph(functions):
    call_graph = {}
    reverse_graph = {}

    # condition-aware edges
    call_edges = {}
    reverse_edges = {}

    for f in functions:
        fname = f["function"]
        call_graph[fname] = f["calls"]

        # condition-aware edges (dedupe by callee + conds + loc)
        edges = []
        seen = set()
        for cs in f.get("callsites", []):
            callee = cs.get("callee")
            conds = cs.get("conds", [])
            loc = cs.get("loc", {})
            key = (callee, " && ".join(conds), loc.get("row"), loc.get("col"))
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "callee": callee,
                "conds": conds,
                "loc": loc,
            })
        call_edges[fname] = edges

        for c in f["calls"]:
            reverse_graph.setdefault(c, []).append(fname)

        for e in edges:
            c = e["callee"]
            reverse_edges.setdefault(c, []).append({
                "caller": fname,
                "conds": e.get("conds", []),
                "loc": e.get("loc"),
            })

    # sort reverse lists for stability
    for k in reverse_graph:
        reverse_graph[k] = sorted(set(reverse_graph[k]))
    for k in reverse_edges:
        reverse_edges[k] = sorted(
            reverse_edges[k],
            key=lambda x: (
                x.get("caller", ""),
                (x.get("loc") or {}).get("row", 10**9),
                (x.get("loc") or {}).get("col", 10**9),
                " && ".join(x.get("conds", [])),
            )
        )

    return call_graph, reverse_graph, call_edges, reverse_edges


# ==========================
# 主程式
# ==========================
def main():
    codebase_root = Path("codebase").resolve()
    all_funcs = []

    for root_dir, _, files in os.walk(codebase_root):
        for file in files:
            if file.endswith(".c"):
                full_path = os.path.join(root_dir, file)
                all_funcs.extend(
                    extract_functions_from_file(full_path, codebase_root)
                )

    call_graph, reverse_graph, call_edges, reverse_edges = build_graph(all_funcs)

    output_data = {
        "function_count": len(all_funcs),
        "functions": all_funcs,
        "call_graph": call_graph,
        "called_by": reverse_graph,
        "call_edges": call_edges,
        "called_by_edges": reverse_edges,
    }

    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/functions.json").write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ AST extraction complete.")
    print("Functions found:", len(all_funcs))
    print("Call graph size:", len(call_graph))
    print("Reverse map size:", len(reverse_graph))
    print("Condition-aware edges:", sum(len(v) for v in call_edges.values()))


if __name__ == "__main__":
    main()