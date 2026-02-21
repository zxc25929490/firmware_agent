import json
import argparse
from pathlib import Path
from collections import deque
import re

def load_graphs(functions_json: Path):
    data = json.loads(functions_json.read_text(encoding="utf-8"))

    # forward: function -> callees
    if "call_graph" in data:
        call_graph = data["call_graph"]
    else:
        call_graph = {}
        for f in data.get("functions", []):
            call_graph[f["function"]] = f.get("calls", [])

    # reverse: callee -> callers
    if "called_by" in data:
        called_by = data["called_by"]
    else:
        called_by = {}
        for caller, callees in call_graph.items():
            for callee in callees:
                called_by.setdefault(callee, []).append(caller)

    return call_graph, called_by

def find_all_paths(graph, root, max_depth=8, max_paths=200, include_pattern=None, exclude_pattern=None):
    inc = re.compile(include_pattern) if include_pattern else None
    exc = re.compile(exclude_pattern) if exclude_pattern else None

    def allowed(name: str) -> bool:
        if exc and exc.search(name):
            return False
        if inc and not inc.search(name):
            return False
        return True

    paths = []
    stack = [(root, [root], set([root]))]

    while stack and len(paths) < max_paths:
        node, path, seen = stack.pop()

        if len(path) >= max_depth:
            paths.append(path + ["…(max_depth)"])
            continue

        children = graph.get(node, [])
        filtered = [c for c in children if allowed(c)]

        if not filtered:
            paths.append(path)
            continue

        for c in reversed(filtered):
            if c in seen:
                paths.append(path + [f"↻{c}"])
                continue
            new_seen = set(seen)
            new_seen.add(c)
            stack.append((c, path + [c], new_seen))

    if len(paths) >= max_paths:
        paths.append([root, f"…(max_paths={max_paths} reached)"])

    return paths

def shortest_path(graph, src, dst, max_nodes=50000):
    if src == dst:
        return [src]
    q = deque([src])
    prev = {src: None}
    visited = set([src])
    nodes = 0

    while q:
        u = q.popleft()
        nodes += 1
        if nodes > max_nodes:
            return None
        for v in graph.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            prev[v] = u
            if v == dst:
                path = [dst]
                cur = dst
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                return list(reversed(path))
            q.append(v)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="output/functions.json")
    ap.add_argument("--func", default=None, help="root function name")
    ap.add_argument("--to", default=None, help="optional target function name (shortest path)")
    ap.add_argument("--up", action="store_true", help="walk upward (caller direction) instead of downward (callee)")
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_paths", type=int, default=200)
    ap.add_argument("--include", default=None, help="regex include filter (display only)")
    ap.add_argument("--exclude", default=None, help="regex exclude filter (display only)")
    args = ap.parse_args()

    call_graph, called_by = load_graphs(Path(args.json))

    root = args.func
    if not root:
        root = input("輸入 function 名稱: ").strip()

    graph = called_by if args.up else call_graph
    direction = "UP (callers)" if args.up else "DOWN (callees)"

    if args.to:
        path = shortest_path(graph, root, args.to)
        if not path:
            print(f"❌ 找不到 path [{direction}]: {root} -> {args.to}")
            return
        print(f"✅ Shortest path [{direction}]:")
        print("  " + " -> ".join(path))
        return

    paths = find_all_paths(
        graph,
        root,
        max_depth=args.max_depth,
        max_paths=args.max_paths,
        include_pattern=args.include,
        exclude_pattern=args.exclude
    )

    print(f"\n🌲 Paths {direction} from {root} (depth<={args.max_depth}, max_paths={args.max_paths})\n")
    for i, p in enumerate(paths, 1):
        print(f"{i:03d}: " + " -> ".join(p))

if __name__ == "__main__":
    main()