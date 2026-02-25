#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser
import tree_sitter_c

# ==========================
# Parser 初始化
# ==========================
parser = Parser()
C_LANGUAGE = Language(tree_sitter_c.language(), "c")
parser.set_language(C_LANGUAGE)

MASK_FUNCS = {"SET_MASK", "CLEAR_MASK"}


# ==========================
# 工具函式
# ==========================
def node_text(src: bytes, node) -> str:
    if node is None:
        return ""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def one_line(s: str, limit: int = 180) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def walk(node):
    yield node
    for ch in node.children:
        yield from walk(ch)


def find_first(node, t: str):
    for n in walk(node):
        if n.type == t:
            return n
    return None


def is_statement_node(n) -> bool:
    return n.type in {
        "if_statement",
        "switch_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "return_statement",
        "break_statement",
        "continue_statement",
        "expression_statement",
        "compound_statement",
        "declaration",
        "labeled_statement",
        "case_statement",
        "default_statement",
    }


# ==========================
# CFG Builder
# ==========================
class CfgBuilder:
    def __init__(self, src: bytes):
        self.src = src
        self.op_id = 0
        self.operations: List[Dict[str, Any]] = []
        self.cfg_edges: List[Dict[str, Any]] = []
        self.var_reads: Dict[str, List[List[str]]] = {}
        self.var_writes: Dict[str, List[List[str]]] = {}

        # loop stack: break/continue target
        self.loop_stack: List[Dict[str, str]] = []

    def _new_id(self) -> str:
        oid = f"op{self.op_id}"
        self.op_id += 1
        return oid

    def add_op(self, op_type: str, text: str, conds: List[str], node) -> str:
        oid = self._new_id()
        row, col = node.start_point
        self.operations.append({
            "id": oid,
            "type": op_type,
            "text": one_line(text),
            "conds": conds[:] if conds else [],
            "loc": {"row": row + 1, "col": col + 1},
        })
        return oid

    def add_edge(self, a: str, b: str, label: str = ""):
        if not a or not b:
            return
        self.cfg_edges.append({"from": a, "to": b, "label": label or ""})

    def record_write(self, var: str, conds: List[str]):
        self.var_writes.setdefault(var, []).append(conds[:])

    def record_read(self, var: str, conds: List[str]):
        self.var_reads.setdefault(var, []).append(conds[:])

    def push_cond(self, conds: List[str], label: str) -> List[str]:
        label = one_line(label)
        return conds + [label] if label else conds

    # -------- expression_statement: 把 assign/call/mask 轉成 ops chain --------
    def analyze_expression_statement(self, expr_stmt, conds: List[str]) -> Tuple[Optional[str], List[str]]:
        ops_chain: List[str] = []

        # assignment
        for n in walk(expr_stmt):
            if n.type == "assignment_expression":
                txt = node_text(self.src, n)
                oid = self.add_op("assign", txt, conds, n)
                ops_chain.append(oid)

                left = n.child_by_field_name("left")
                if left and left.type == "identifier":
                    self.record_write(node_text(self.src, left), conds)

        # calls
        for n in walk(expr_stmt):
            if n.type == "call_expression":
                fn_node = n.child_by_field_name("function")
                fn_name = node_text(self.src, fn_node) if fn_node else ""
                txt = node_text(self.src, n)

                if fn_name in MASK_FUNCS:
                    oid = self.add_op("mask_op", txt, conds, n)
                else:
                    oid = self.add_op("call", txt, conds, n)
                ops_chain.append(oid)

        # identifier reads (rough)
        assign_left_nodes = set()
        for n in walk(expr_stmt):
            if n.type == "assignment_expression":
                left = n.child_by_field_name("left")
                if left is not None:
                    assign_left_nodes.add((left.start_byte, left.end_byte))

        for n in walk(expr_stmt):
            if n.type == "identifier":
                key = (n.start_byte, n.end_byte)
                if key in assign_left_nodes:
                    continue
                self.record_read(node_text(self.src, n), conds)

        if not ops_chain:
            return None, []

        for a, b in zip(ops_chain, ops_chain[1:]):
            self.add_edge(a, b, "seq")

        return ops_chain[0], [ops_chain[-1]]

    # -------- stmt builder: return (entry_id, exits, terminal) --------
    def build_stmt(self, stmt, conds: List[str]) -> Tuple[Optional[str], List[str], bool]:
        if stmt is None:
            return None, [], False

        t = stmt.type

        # compound block
        if t == "compound_statement":
            return self.build_block(stmt, conds)

        # if
        if t == "if_statement":
            cond_node = stmt.child_by_field_name("condition")
            cond_txt = one_line(node_text(self.src, cond_node)) if cond_node else "<cond>"
            if_id = self.add_op("if", f"if({cond_txt})", conds, stmt)
            join_id = self.add_op("join", "join_if", conds, stmt)

            cons = stmt.child_by_field_name("consequence")
            alt = stmt.child_by_field_name("alternative")

            cons_entry, cons_exits, cons_terminal = self.build_stmt(cons, self.push_cond(conds, f"if({cond_txt})"))
            if cons_entry:
                self.add_edge(if_id, cons_entry, "T")
                if not cons_terminal:
                    for ex in cons_exits:
                        self.add_edge(ex, join_id, "join")
            else:
                self.add_edge(if_id, join_id, "T")

            if alt:
                alt_entry, alt_exits, alt_terminal = self.build_stmt(alt, self.push_cond(conds, f"else({cond_txt})"))
                if alt_entry:
                    self.add_edge(if_id, alt_entry, "F")
                    if not alt_terminal:
                        for ex in alt_exits:
                            self.add_edge(ex, join_id, "join")
                else:
                    self.add_edge(if_id, join_id, "F")
            else:
                self.add_edge(if_id, join_id, "F")

            return if_id, [join_id], False

        # switch
        if t == "switch_statement":
            val_node = stmt.child_by_field_name("condition")
            val_txt = one_line(node_text(self.src, val_node)) if val_node else "<expr>"
            sw_id = self.add_op("switch", f"switch({val_txt})", conds, stmt)

            body = stmt.child_by_field_name("body")
            if not body:
                return sw_id, [], False

            all_terminal = True

            # 直接處理 case_statement
            for ch in body.children:
                if ch.type in ("case_statement", "default_statement"):

                    if ch.type == "case_statement":
                        v = ch.child_by_field_name("value")
                        label = f"case({one_line(node_text(self.src, v))})"
                    elif ch.type == "default_statement":
                        label = "default"

                    case_id = self.add_op("case", label, conds, ch)
                    self.add_edge(sw_id, case_id, label)

                    # 🔥 直接對整個 case_statement build
                    entry, exits, terminal = self.build_stmt(ch, conds)

                    if entry:
                        self.add_edge(case_id, entry, "enter")

                    if not terminal:
                        all_terminal = False

            # 如果全部 case 都 return → 整個 switch terminal
            if all_terminal:
                return sw_id, [], True

            return sw_id, [], False

        # loops
        if t in ("while_statement", "for_statement", "do_statement"):
            loop_txt = one_line(node_text(self.src, stmt))
            loop_id = self.add_op("loop", loop_txt, conds, stmt)
            join_id = self.add_op("join", "join_loop", conds, stmt)

            self.loop_stack.append({"continue_to": loop_id, "break_to": join_id})
            body = stmt.child_by_field_name("body")
            body_entry, body_exits, body_terminal = self.build_stmt(body, self.push_cond(conds, f"loop({loop_txt})"))
            self.loop_stack.pop()

            if body_entry:
                self.add_edge(loop_id, body_entry, "T")
                if not body_terminal:
                    for ex in body_exits:
                        self.add_edge(ex, loop_id, "back")
            else:
                self.add_edge(loop_id, loop_id, "back")

            self.add_edge(loop_id, join_id, "F")
            return loop_id, [join_id], False

        # break / continue
        if t == "break_statement":
            br_id = self.add_op("break", "break", conds, stmt)
            if self.loop_stack:
                self.add_edge(br_id, self.loop_stack[-1]["break_to"], "break_to")
            return br_id, [], True

        if t == "continue_statement":
            ct_id = self.add_op("continue", "continue", conds, stmt)
            if self.loop_stack:
                self.add_edge(ct_id, self.loop_stack[-1]["continue_to"], "continue_to")
            return ct_id, [], True

        # return
        if t == "return_statement":
            txt = node_text(self.src, stmt)
            rt_id = self.add_op("return", txt, conds, stmt)
            for n in walk(stmt):
                if n.type == "identifier":
                    self.record_read(node_text(self.src, n), conds)
            return rt_id, [], True

        # expression statement
        if t == "expression_statement":
            entry, exits = self.analyze_expression_statement(stmt, conds)
            if entry is None:
                return None, [], False
            return entry, exits, False

        # declaration
        if t == "declaration":
            txt = node_text(self.src, stmt)
            dec_id = self.add_op("decl", txt, conds, stmt)

            first_id = find_first(stmt, "identifier")
            if first_id is not None:
                self.record_write(node_text(self.src, first_id), conds)

            for n in walk(stmt):
                if n.type == "identifier":
                    if first_id is not None and n.start_byte == first_id.start_byte and n.end_byte == first_id.end_byte:
                        continue
                    self.record_read(node_text(self.src, n), conds)

            return dec_id, [dec_id], False

        # generic: try build children statements in order
        child_stmts = [ch for ch in stmt.children if is_statement_node(ch)]
        return self.build_stmt_list(child_stmts, conds)

    # ==========================
    # ✅ 這裡是「最關鍵修正」：遇到 terminal 立刻結束 block
    # ==========================
    def build_stmt_list(self, stmts: List[Any], conds: List[str]) -> Tuple[Optional[str], List[str], bool]:
        entry: Optional[str] = None
        prev_exits: List[str] = []

        for st in stmts:
            st_entry, st_exits, st_terminal = self.build_stmt(st, conds)

            if st_entry is None and not st_terminal:
                continue

            if entry is None and st_entry:
                entry = st_entry

            if prev_exits and st_entry:
                for ex in prev_exits:
                    self.add_edge(ex, st_entry, "seq")

            if st_terminal:
                # ✅ 一旦 terminal，整段 block 結束（不要再把後面 stmt 串起來）
                return entry, [], True

            prev_exits = st_exits

        return entry, prev_exits, False

    def build_block(self, block_node, conds: List[str]) -> Tuple[Optional[str], List[str], bool]:
        stmts = [ch for ch in block_node.children if is_statement_node(ch)]
        return self.build_stmt_list(stmts, conds)


# ==========================
# Extract：每檔每 function
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
        if not body:
            continue

        b = CfgBuilder(src)
        entry_id = b.add_op("entry", f"ENTRY {func_name}", [], node)

        body_entry, body_exits, body_terminal = b.build_stmt(body, [])
        if body_entry:
            b.add_edge(entry_id, body_entry, "enter")

        end_id = b.add_op("end", f"END {func_name}", [], node)
        if not body_terminal:
            for ex in body_exits:
                b.add_edge(ex, end_id, "end")

        calls = sorted({
            op["text"].split("(")[0].strip()
            for op in b.operations
            if op["type"] in {"call", "mask_op"}
        })

        results.append({
            "function": func_name,
            "file": str(Path(path).relative_to(codebase_root)).replace("\\", "/"),
            "calls": calls,
            "operations": b.operations,
            "cfg_edges": b.cfg_edges,
            "variable_reads": b.var_reads,
            "variable_writes": b.var_writes,
        })

    return results


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
                all_funcs.extend(extract_functions_from_file(full_path, codebase_root))

    output_data = {
        "function_count": len(all_funcs),
        "functions": all_funcs,
    }

    Path("output").mkdir(parents=True, exist_ok=True)
    Path("output/functions.json").write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("✅ CFG AST extraction complete.")
    print("Functions found:", len(all_funcs))


if __name__ == "__main__":
    main()