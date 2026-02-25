#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
query_call.py

Firmware Call Graph Query Engine
---------------------------------
Importable graph engine for firmware agent.

Usage:

    from query_call import FirmwareQueryEngine

    engine = FirmwareQueryEngine("output/functions.json")

    paths = engine.paths_down("Service_CPU_Thermal_Table", max_depth=8)
    print(paths)

"""

import json
import re
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


# ==========================================================
# Core Graph Engine
# ==========================================================

class FirmwareQueryEngine:

    def __init__(self, json_path: str = "output/functions.json"):
        self.json_path = Path(json_path)
        self.call_graph = {}
        self.called_by = {}
        self.call_edges = {}
        self.called_by_edges = {}
        self.func_names = []
        self._load_graph()

    # ------------------------------------------------------
    # Load Graph
    # ------------------------------------------------------

    def _load_graph(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"{self.json_path} not found")

        data = json.loads(self.json_path.read_text(encoding="utf-8"))

        self.call_graph = data.get("call_graph", {})
        self.called_by = data.get("called_by", {})
        self.call_edges = data.get("call_edges")
        self.called_by_edges = data.get("called_by_edges")

        self.func_names = [
            f.get("function")
            for f in data.get("functions", [])
            if f.get("function")
        ]

        # fallback reverse graph
        if not self.called_by:
            for caller, callees in self.call_graph.items():
                for callee in callees:
                    self.called_by.setdefault(callee, []).append(caller)

    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------

    def list_functions(self, grep: Optional[str] = None) -> List[str]:
        names = sorted(set(self.func_names))
        if grep:
            pattern = re.compile(grep)
            names = [n for n in names if pattern.search(n)]
        return names

    def _iter_children(
        self,
        graph: Dict[str, Any],
        edges_graph: Optional[Dict[str, Any]],
        node: str,
    ) -> List[Tuple[str, List[str]]]:

        result = []

        if edges_graph and node in edges_graph:
            for e in edges_graph[node]:
                child = e.get("callee") or e.get("caller")
                conds = e.get("conds", []) or []
                if child:
                    result.append((child, conds))
            return result

        for c in graph.get(node, []) or []:
            result.append((c, []))

        return result

    # ------------------------------------------------------
    # DFS Path Enumeration
    # ------------------------------------------------------

    def _find_paths(
        self,
        graph,
        edges_graph,
        root,
        max_depth=10,
        max_paths=200,
    ):

        paths = []
        stack = [(root, [root], {root})]

        while stack and len(paths) < max_paths:
            node, path, seen = stack.pop()

            if len(path) >= max_depth:
                paths.append(path + ["…(max_depth)"])
                continue

            children = self._iter_children(graph, edges_graph, node)

            if not children:
                paths.append(path)
                continue

            for c, _ in reversed(children):
                if c in seen:
                    paths.append(path + [f"↻{c}"])
                    continue
                new_seen = set(seen)
                new_seen.add(c)
                stack.append((c, path + [c], new_seen))

        return paths

    # ------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------

    def paths_down(self, root: str, max_depth=10, max_paths=200):
        return self._find_paths(
            self.call_graph,
            self.call_edges,
            root,
            max_depth=max_depth,
            max_paths=max_paths,
        )

    def paths_up(self, root: str, max_depth=10, max_paths=200):
        return self._find_paths(
            self.called_by,
            self.called_by_edges,
            root,
            max_depth=max_depth,
            max_paths=max_paths,
        )

    # ------------------------------------------------------
    # Shortest Path (BFS)
    # ------------------------------------------------------

    def shortest_path(self, src: str, dst: str, max_depth=200):

        if src == dst:
            return [src]

        q = deque([(src, 0)])
        visited = {src}
        prev = {src: None}

        while q:
            node, depth = q.popleft()

            if depth >= max_depth:
                continue

            for child, _ in self._iter_children(
                self.call_graph,
                self.call_edges,
                node
            ):
                if child in visited:
                    continue

                visited.add(child)
                prev[child] = node

                if child == dst:
                    path = [dst]
                    cur = dst
                    while prev[cur] is not None:
                        cur = prev[cur]
                        path.append(cur)
                    return list(reversed(path))

                q.append((child, depth + 1))

        return None

    # ------------------------------------------------------
    # Sink Search
    # ------------------------------------------------------

    def find_sinks(
        self,
        root: str,
        sink_regex: str,
        max_depth=10,
        max_paths=200
    ):

        pattern = re.compile(sink_regex)
        paths = self.paths_down(root, max_depth, max_paths)

        result = []
        for p in paths:
            for node in p:
                if pattern.search(node):
                    result.append({
                        "sink": node,
                        "path": p
                    })
                    break

        return result