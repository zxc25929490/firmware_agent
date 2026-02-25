#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests

from query_call import FirmwareQueryEngine


# =========================
# Config
# =========================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("FIRMWARE_MODEL", "qwen2.5:14b-instruct")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "codebase")

CALL_GRAPH_JSON = os.environ.get("CALL_GRAPH_JSON", "output/functions.json")

MAX_CHROMA_HITS = 6
MAX_WRITES = 12
MAX_CLEARS = 12
MAX_TOP_ENTRY = 5
MAX_PATHS_SHOW = 2

# =========================
# Graph
# =========================
GRAPH = FirmwareQueryEngine(CALL_GRAPH_JSON)
FUNC_SET = set(getattr(GRAPH, "func_names", []) or [])

# =========================
# LLM System Prompt
# =========================
ZH_SYSTEM = """你是資深 Firmware/EC/BIOS 除錯工程師。

【硬規則】
- 一律使用「繁體中文」輸出。
- function/變數/巨集/寄存器名稱保持原樣，不翻譯。
- 嚴禁把 ON/OFF/TRUE/FALSE/ENABLE/DISABLE/Module_OFF 這類狀態詞當成 function 或證據。
- 嚴禁在沒有 repo 證據時，提出「driver/OS/硬體故障」作為根因。除非 Evidence Digest 明確提到 OS driver/hardware/power rail。
- 若證據不足：要明講「缺什麼證據」，並提出可操作的下一步（要哪段 log / 要哪個 reg dump / 要搜什麼 pattern）。

【引用規則（Evidence-Gated）】
- 我會提供 Evidence Digest，裡面每條都有編號 [E1] [E2]...
- 你在根因/推論時，必須引用對應的 evidence 編號（至少 1 個）。
- 不得引用不存在的編號；若沒有能支撐的 evidence，就必須說「目前沒有直接證據」。

【工程工作方式】
- 你可以提出假設，但必須對應到 Evidence 或清楚說明缺口與怎麼補證據。
"""


STOP_SYMBOLS = {
    "ON", "OFF", "TRUE", "FALSE",
    "ENABLE", "DISABLE",
    "HIGH", "LOW",
    "Module_ON", "Module_OFF",
}


# =========================
# Utils
# =========================
def _english_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    return letters / max(1, len(text))


def ask_llm_raw(prompt: str, temperature: float = 0.2, timeout_s: int = 180) -> str:
    full = ZH_SYSTEM + "\n\n" + prompt.strip() + "\n"
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout_s,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def ask_llm(prompt: str, temperature: float = 0.2) -> str:
    out = ask_llm_raw(prompt, temperature=temperature)
    # 英文比例過高 → 二次改寫成繁中
    if _english_ratio(out) > 0.10:
        rewrite = f"""
請把下列內容「完整改寫成繁體中文」，保留所有 function/變數/巨集/寄存器名稱原樣，不要加入新資訊，不要刪掉重點：

--- 原文開始 ---
{out}
--- 原文結束 ---
"""
        out2 = ask_llm_raw(rewrite, temperature=0.1)
        if out2:
            return out2.strip()
    return out


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(CHROMA_COLLECTION)


# =========================
# Chroma search
# =========================
def chroma_query(query: str, n: int = 5) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    col = get_collection()
    res = col.query(query_texts=[q], n_results=n)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        hits.append({
            "query": q,
            "function": meta.get("function", "UNKNOWN_FN"),
            "path": meta.get("path", meta.get("file", "UNKNOWN_PATH")),
            "distance": float(dist) if dist is not None else 9999.0,
            "excerpt": "\n".join((doc or "").splitlines()[:10]).strip() if isinstance(doc, str) else "",
        })
    hits.sort(key=lambda x: x["distance"])
    return hits


def chroma_query_many(queries: List[str], n_each: int = 4) -> List[Dict[str, Any]]:
    all_hits = []
    for q in queries:
        all_hits.extend(chroma_query(q, n=n_each))

    # dedup
    seen = set()
    dedup = []
    for h in sorted(all_hits, key=lambda x: x["distance"]):
        key = (h["function"], h["path"], (h.get("excerpt") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)
    return dedup[:MAX_CHROMA_HITS]


def chroma_get_all_docs() -> Dict[str, Any]:
    col = get_collection()
    return col.get()


# =========================
# Intent Detection
# =========================
def detect_mode(q: str) -> str:
    ql = (q or "").lower()

    # function explain
    if any(k in ql for k in ["在做甚麼", "做什麼", "用途", "解釋", "功能", "幹嘛", "做啥"]):
        return "function"

    # variable trace
    if any(k in ql for k in ["哪裡改", "哪裡被改", "誰改", "誰寫", "哪裡設", "被改", "被設", "writer", "trace"]):
        return "trace"

    # debug + must use code evidence
    if any(k in ql for k in ["從code", "從 code", "從程式碼", "code中", "code 中", "根據code", "用code分析", "repo中分析", "從repo分析"]):
        return "debug_code"

    return "debug"


# =========================
# Symbol Extraction
# =========================
def extract_symbols(question: str, max_syms: int = 10) -> Dict[str, List[str]]:
    text = question or ""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    tokens = [t for t in tokens if len(t) >= 3]

    funcs = []
    vars_macros = []
    regs = []

    for t in tokens:
        if t in STOP_SYMBOLS:
            continue

        # function known by call graph
        if t in FUNC_SET:
            funcs.append(t)
            continue

        # register-ish
        if t.isupper() and any(k in t for k in ("REG", "CTRL", "CFG", "STAT", "STS", "GPIO", "I2C", "SMB", "EC_RAM")):
            regs.append(t)
            continue

        # vars/macros
        if "_" in t or (not t.isupper() and any(c.isupper() for c in t[1:])):
            vars_macros.append(t)
            continue

        if t.isupper():
            vars_macros.append(t)

    def uniq(xs):
        out, seen = [], set()
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {
        "functions": uniq(funcs)[:8],
        "vars_macros": uniq(vars_macros)[:max_syms],
        "registers": uniq(regs)[:8],
    }


# =========================
# Writer / Clear scan (regex)
# =========================
def scan_writer_clear(symbol: str, max_hits: int = 200) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not symbol or symbol in STOP_SYMBOLS:
        return [], []

    res = chroma_get_all_docs()
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []

    # 支援常見寫入 / bitwise / macro pattern（不完美，但比只抓 '=' 強很多）
    pat_writer = re.compile(
        rf"(?mi)^\s*.*\b{re.escape(symbol)}\b\s*(=|\+=|-=|\|=|&=|\^=|<<=|>>=)\s*.*$"
    )
    pat_macro_writer = re.compile(
        rf"(?mi)^\s*.*(SET|CLR|CLEAR|RESET|WRITE|UPDATE)_[A-Z0-9_]*\s*\(\s*{re.escape(symbol)}\b.*$"
    )
    pat_clear = re.compile(
        rf"(?mi)^\s*.*\b{re.escape(symbol)}\b.*(?:=\s*0\b|&=\s*~|\bCLEAR\b|\bRESET\b|\bDISABLE\b|\bCLR\b).*?$"
    )

    writers: List[Dict[str, str]] = []
    clears: List[Dict[str, str]] = []

    for meta, doc in zip(metas, docs):
        if not isinstance(doc, str):
            continue
        fn = (meta or {}).get("function", "UNKNOWN_FN")

        for m in pat_writer.finditer(doc):
            line = re.sub(r"\s+", " ", m.group(0)).strip()
            writers.append({"function": fn, "excerpt": f"[{symbol}] {line}"})
            if len(writers) >= max_hits:
                break

        for m in pat_macro_writer.finditer(doc):
            line = re.sub(r"\s+", " ", m.group(0)).strip()
            writers.append({"function": fn, "excerpt": f"[{symbol}] {line}"})
            if len(writers) >= max_hits:
                break

        for m in pat_clear.finditer(doc):
            line = re.sub(r"\s+", " ", m.group(0)).strip()
            clears.append({"function": fn, "excerpt": f"[{symbol}] {line}"})
            if len(clears) >= max_hits:
                break

        if len(writers) >= max_hits and len(clears) >= max_hits:
            break

    # dedup
    def dedup(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out, seen = [], set()
        for it in items:
            k = (it["function"], it["excerpt"])
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    return dedup(writers)[:MAX_WRITES], dedup(clears)[:MAX_CLEARS]


def rank_entrypoints_from_writers(writers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    writer_funcs = [w["function"] for w in writers if w.get("function") in FUNC_SET]
    if not writer_funcs:
        return []

    counter = Counter()
    entry_to_writers = defaultdict(set)
    example_paths = defaultdict(list)

    for wf in writer_funcs:
        ups = GRAPH.paths_up(wf, max_depth=10, max_paths=200) or []
        if not ups:
            entry = wf
            counter[entry] += 1
            entry_to_writers[entry].add(wf)
            if not example_paths[entry]:
                example_paths[entry].append([wf])
            continue

        for p in ups:
            if not p:
                continue
            entry = p[-1]
            counter[entry] += 1
            entry_to_writers[entry].add(wf)
            if len(example_paths[entry]) < MAX_PATHS_SHOW:
                example_paths[entry].append(p)

    ranked = []
    for entry, cnt in counter.most_common(MAX_TOP_ENTRY):
        ranked.append({
            "entry": entry,
            "paths_count": cnt,
            "writers": sorted(list(entry_to_writers[entry]))[:8],
            "examples": example_paths[entry][:MAX_PATHS_SHOW],
        })
    return ranked


# =========================
# Evidence Builder (Generic)
# =========================
def build_evidence(question: str) -> Dict[str, Any]:
    syms = extract_symbols(question)
    # queries: symbols + original question (fallback semantic)
    queries = []
    for s in (syms["vars_macros"] + syms["functions"] + syms["registers"])[:10]:
        queries.append(s)
        queries.append(f"{s} clear")
        queries.append(f"{s} reset")
    queries.append(question)

    hits = chroma_query_many(queries[:18], n_each=3)

    related_fns = []
    for h in hits:
        fn = h.get("function")
        if fn and fn != "UNKNOWN_FN":
            related_fns.append(fn)
    related_fns = list(dict.fromkeys(related_fns))[:6]

    callgraph = []
    for fn in related_fns[:4]:
        ups = GRAPH.paths_up(fn, max_depth=4, max_paths=8) or []
        downs = GRAPH.paths_down(fn, max_depth=4, max_paths=8) or []
        callgraph.append({"fn": fn, "up": ups[:2], "down": downs[:2]})

    writers_all = []
    clears_all = []
    # 掃 vars/macros（最有機會是 flag / 狀態）
    for sym in (syms["vars_macros"] or [])[:5]:
        w, c = scan_writer_clear(sym)
        writers_all.extend(w)
        clears_all.extend(c)

    top_entry = rank_entrypoints_from_writers(writers_all)

    return {
        "symbols": syms,
        "hits": hits,
        "related_fns": related_fns,
        "callgraph": callgraph,
        "writers": writers_all[:MAX_WRITES],
        "clears": clears_all[:MAX_CLEARS],
        "top_entry": top_entry[:MAX_TOP_ENTRY],
    }


def format_digest(ev: Dict[str, Any]) -> str:
    sym = ev.get("symbols") or {}
    hits = ev.get("hits") or []
    writers = ev.get("writers") or []
    clears = ev.get("clears") or []
    top_entry = ev.get("top_entry") or []
    callgraph = ev.get("callgraph") or []

    eid = 1
    lines = []

    lines.append("【抽到的 symbols】")
    lines.append(f"- functions: {', '.join(sym.get('functions', []) or []) or '(無)'}")
    lines.append(f"- vars/macros: {', '.join(sym.get('vars_macros', []) or []) or '(無)'}")
    lines.append(f"- registers: {', '.join(sym.get('registers', []) or []) or '(無)'}")
    lines.append("")

    lines.append("【Chroma 命中（Top）】")
    if not hits:
        lines.append(f"[E{eid}] (無命中)"); eid += 1
    else:
        for h in hits[:MAX_CHROMA_HITS]:
            lines.append(f"[E{eid}] fn={h['function']} | dist={h['distance']:.4f} | file={h['path']}")
            eid += 1
            if h.get("excerpt"):
                for ln in h["excerpt"].splitlines()[:3]:
                    lines.append(f"    {ln}")
    lines.append("")

    lines.append("【Call Graph（上下游）】")
    if not callgraph:
        lines.append(f"[E{eid}] (無)"); eid += 1
    else:
        for cg in callgraph[:4]:
            lines.append(f"[E{eid}] fn={cg['fn']}")
            eid += 1
            for p in cg.get("up", [])[:2]:
                lines.append(f"    up: {' -> '.join(p)}")
            for p in cg.get("down", [])[:2]:
                lines.append(f"    down: {' -> '.join(p)}")
    lines.append("")

    lines.append("【writer（Top）】")
    if not writers:
        lines.append(f"[E{eid}] (沒有掃到賦值/位元操作寫入點)"); eid += 1
    else:
        for w in writers[:MAX_WRITES]:
            lines.append(f"[E{eid}] {w['function']}: {w['excerpt']}")
            eid += 1
    lines.append("")

    lines.append("【clear/reset（Top）】")
    if not clears:
        lines.append(f"[E{eid}] (沒有掃到明顯 clear/reset pattern)"); eid += 1
    else:
        for c in clears[:MAX_CLEARS]:
            lines.append(f"[E{eid}] {c['function']}: {c['excerpt']}")
            eid += 1
    lines.append("")

    lines.append("【writer -> paths_up Top Entrypoint】")
    if not top_entry:
        lines.append(f"[E{eid}] (writer function 不在 call graph 或找不到 paths_up)"); eid += 1
    else:
        for i, e in enumerate(top_entry[:MAX_TOP_ENTRY], 1):
            lines.append(f"[E{eid}] {i}. 入口：{e['entry']} | writers={', '.join(e.get('writers', []) or []) or '(無)'} | paths_count={e.get('paths_count', 0)}")
            eid += 1
            for p in (e.get("examples") or [])[:MAX_PATHS_SHOW]:
                lines.append(f"    - 例： {' -> '.join(p)}")

    return "\n".join(lines)


# =========================
# Mode Handlers
# =========================
def function_mode(question: str) -> str:
    # 抓第一個 symbol 當 function
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question or "")
    fn = tokens[0] if tokens else ""
    if not fn:
        return "我沒有抓到 function 名稱。請用像「HotKey_Fn_F12 在做甚麼」這種問法。"

    hits = chroma_query(fn, n=3)
    code_excerpt = hits[0]["excerpt"] if hits else ""

    callers = GRAPH.paths_up(fn, max_depth=4, max_paths=10) or []
    callees = GRAPH.paths_down(fn, max_depth=4, max_paths=10) or []

    prompt = f"""
想了解的 function：{fn}

Code excerpt（可能不完整）：
{code_excerpt or "(Chroma 未命中或 excerpt 為空)"}

Callers（paths_up）：
{callers}

Callees（paths_down）：
{callees}

請輸出：
1) 功能說明（它在做什麼）
2) 主要邏輯流程（用條列）
3) 呼叫關係（上游/下游）
4) 可能的副作用（改哪些 flag/狀態/寄存器）
如果 excerpt 不足，請明講「缺哪段 code」，並建議我該搜尋哪個關鍵字或要哪個檔案片段。
"""
    return ask_llm(prompt, temperature=0.2)


def trace_mode(question: str) -> str:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question or "")
    sym = tokens[0] if tokens else ""
    if not sym:
        return "我沒有抓到變數/巨集名稱。請用像「Hotkey_CAMERA 哪裡改」這種問法。"

    w, c = scan_writer_clear(sym)
    top_entry = rank_entrypoints_from_writers(w)

    # 額外補：用 Chroma 找更多上下文
    hits = chroma_query_many([sym, f"{sym} clear", f"{sym} reset"], n_each=4)

    digest = []
    digest.append(f"變數/巨集：{sym}")
    digest.append("\n【writers】")
    digest += [f"- {x['function']}: {x['excerpt']}" for x in w] if w else ["- (無)"]
    digest.append("\n【clears/resets】")
    digest += [f"- {x['function']}: {x['excerpt']}" for x in c] if c else ["- (無)"]
    digest.append("\n【Top Entrypoint（由 writers 反推）】")
    if top_entry:
        for e in top_entry:
            digest.append(f"- {e['entry']} | writers={', '.join(e.get('writers', []))}")
            for p in (e.get("examples") or [])[:MAX_PATHS_SHOW]:
                digest.append(f"  例：{' -> '.join(p)}")
    else:
        digest.append("- (無)")

    digest.append("\n【Chroma 命中】")
    if hits:
        for h in hits[:MAX_CHROMA_HITS]:
            digest.append(f"- fn={h['function']} | file={h['path']} | dist={h['distance']:.4f}")
    else:
        digest.append("- (無)")

    prompt = f"""
使用者想追：{sym}

我掃到的證據：
{chr(10).join(digest)}

請輸出：
1) 最可能的「寫入點」是哪些（優先列出 writers，有 excerpt 就引用）
2) 是否存在 override/定時器覆寫風險（若有 entrypoint 是 timer hook，要講明）
3) 下一步怎麼驗證（在哪些 function 加 log、印哪些值、要抓哪段 log）
限制：不要靠 function 名稱硬猜，請盡量引用上面掃到的 excerpt 作為證據。
"""
    return ask_llm(prompt, temperature=0.2)


def debug_code_mode(question: str) -> str:
    ev = build_evidence(question)
    digest = format_digest(ev)

    has_real_evidence = bool(ev["hits"] or ev["writers"] or ev["clears"] or ev["callgraph"])
    if not has_real_evidence:
        return (
            "目前資料庫內沒有足夠 repo 證據可以「從 code 分析」。\n"
            "你可以補其中一項，我就能繼續：\n"
            "1) 你的 log 片段（含關鍵字，例如 function 名 / reg 名 / 變數名）\n"
            "2) 你懷疑的變數/bit/寄存器名稱（例如 EC_RAM82 / GPIOB2 / HotKeyStatus）\n"
            "3) 你觀察到的觸發事件（timer? interrupt? HID write? WMI?）"
        )

    prompt = f"""
使用者問題：{question}

Evidence Digest（只能引用 [E#] 作為證據）：
{digest}

請輸出（必須繁中）：
1) Top3 最可能根因（每個：為什麼 + 至少引用 1 個 [E#]）
2) 下一步 Debug Plan（具體：在哪些 function 加 log、要印哪些變數/bit、或要抓哪段 log/reg dump）
3) 2~3 個關鍵追問（用來縮小範圍）

限制：
- 嚴禁在沒有 Evidence 支撐下，講 driver/硬體故障/OS 問題。
- 若 Evidence 沒辦法直接指向根因，請明講「目前沒有直接證據」，並提出如何補證據的具體步驟。
"""
    return ask_llm(prompt, temperature=0.2)


def debug_mode(question: str) -> str:
    # 一般 debug：允許比較粗，但仍建議引用證據；此 mode 不強制 evidence gating
    # 但我仍然會先嘗試 build_evidence，讓它不要亂講
    ev = build_evidence(question)
    digest = format_digest(ev)

    prompt = f"""
使用者問題：{question}

（我有做 repo 搜尋，結果如下；如果你能引用 [E#] 請引用）
Evidence Digest：
{digest}

請輸出：
1) 你目前的初步判斷（可含假設，但要說明信心）
2) 最值得先查的 3 個方向（要具體到 function/變數/檔案）
3) 下一步要我提供什麼資料（log/reg/spec/觸發條件）
"""
    return ask_llm(prompt, temperature=0.2)


# =========================
# Memory (light)
# =========================
DEFAULT_MEM = {"history": [], "last_mode": None, "last_symbols": {}}

def load_mem(path: Optional[str]) -> Dict[str, Any]:
    mem = json.loads(json.dumps(DEFAULT_MEM))
    if not path:
        return mem
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                mem.update(data)
        except Exception:
            pass
    return mem

def save_mem(path: Optional[str], mem: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-q", "--question", default=None)
    parser.add_argument("--memory-file", default=None)
    args = parser.parse_args()

    mem = load_mem(args.memory_file)

    def handle(q: str) -> str:
        mode = detect_mode(q)
        mem["last_mode"] = mode
        mem["history"].append({"q": q[:400], "mode": mode})
        if len(mem["history"]) > 40:
            mem["history"] = mem["history"][-40:]

        if mode == "function":
            return function_mode(q)
        if mode == "trace":
            return trace_mode(q)
        if mode == "debug_code":
            return debug_code_mode(q)
        return debug_mode(q)

    if args.interactive:
        while True:
            q = input("firmware> ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            if q.lower() == "state":
                print(json.dumps({
                    "model": MODEL,
                    "ollama_url": OLLAMA_URL,
                    "chroma_path": CHROMA_PATH,
                    "collection": CHROMA_COLLECTION,
                    "call_graph": CALL_GRAPH_JSON,
                    "last_mode": mem.get("last_mode"),
                }, ensure_ascii=False, indent=2))
                continue
            if q.lower() == "reset":
                mem = json.loads(json.dumps(DEFAULT_MEM))
                print("已 reset（清空 memory）")
                save_mem(args.memory_file, mem)
                continue

            try:
                out = handle(q)
                print(out)
            except Exception as e:
                print(f"ERROR: {e}")

            save_mem(args.memory_file, mem)

    else:
        if not args.question:
            print("請用 -q 提供問題，或用 -i 互動模式")
            return
        out = handle(args.question)
        print(out)
        save_mem(args.memory_file, mem)


if __name__ == "__main__":
    main()