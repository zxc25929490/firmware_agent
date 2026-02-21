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
def node_text(src, node):
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


# ==========================
# 核心抽取
# ==========================
def extract_functions_from_file(path, codebase_root):
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
        calls = []

        if body:
            for sub in walk(body):
                if sub.type == "call_expression":
                    fn_node = sub.child_by_field_name("function")
                    if fn_node:
                        calls.append(node_text(src, fn_node))

        results.append({
            "function": func_name,
            "file": str(Path(path).relative_to(codebase_root)).replace("\\", "/"),
            "calls": sorted(set(calls))
        })

    return results


# ==========================
# 建 graph + reverse map
# ==========================
def build_graph(functions):
    call_graph = {}
    reverse_graph = {}

    for f in functions:
        fname = f["function"]
        call_graph[fname] = f["calls"]

        for c in f["calls"]:
            reverse_graph.setdefault(c, []).append(fname)

    return call_graph, reverse_graph


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

    call_graph, reverse_graph = build_graph(all_funcs)

    output_data = {
        "function_count": len(all_funcs),
        "functions": all_funcs,
        "call_graph": call_graph,
        "called_by": reverse_graph
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


if __name__ == "__main__":
    main()