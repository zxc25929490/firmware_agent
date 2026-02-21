#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Firmware Analysis Engine v3
Modes:
  - Explain Mode: function semantic (reads real C code from Chroma)
  - Control Flow Mode: decision graph reasoning (structured JSON)
  - Sink Trace Mode: who calls sink (callers)
  - Data Flow Mode: mask/flag/bit/assign analysis (reads real C code + optional grep callers)
"""

import argparse
import json
import re
from pathlib import Path
import requests
import chromadb

# =========================
# Config
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

CHROMA_PATH = "./chroma_db"
COLLECTION = "codebase"

# Your existing structured decision graph (LED cases etc.)
STRUCTURED_JSON = "output/flow_v2/led_cases_lite.json"

# =========================
# Ollama
# =========================

def ask_llm(prompt, temperature=0.2):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]

# =========================
# Chroma
# =========================

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION)

def get_function_code(fn: str):
    col = get_collection()
    res = col.get(where={"function": fn})
    if res.get("documents"):
        return res["documents"][0]
    return None

def find_callers(target_fn: str):
    """
    Heuristic caller search:
    - scan all documents in collection
    - treat 'target_fn(' as callsite (ignore comments best-effort)
    """
    col = get_collection()
    res = col.get()
    callers = []

    # match target_fn( with optional spaces, avoid matching in identifiers like foo_target_fn_bar
    call_pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(target_fn)}\s*\(")

    for meta, doc in zip(res.get("metadatas", []), res.get("documents", [])):
        if not isinstance(doc, str):
            continue
        if call_pat.search(doc):
            callers.append(meta.get("function", "UNKNOWN_FN"))

    return sorted(set(callers))

# =========================
# Helpers: function name parsing
# =========================

STOPWORDS = {
    "firmware", "cpu", "fan", "update", "code", "flow", "mask",
    "在哪裡", "哪裡", "有", "設定", "set", "the", "a", "an", "to", "of",
    "this", "that", "is", "are"
}

def extract_function_name(question: str):
    """
    Try to find a plausible C identifier that exists in Chroma.
    Strategy:
      1) collect all identifier-like tokens
      2) prefer tokens that exist as a function in Chroma
      3) fallback to last identifier token
    """
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    toks = [t for t in toks if t.lower() not in STOPWORDS]

    if not toks:
        return None

    col = get_collection()
    # try direct hits
    for t in toks:
        res = col.get(where={"function": t})
        if res.get("documents"):
            return t

    # fallback: last token that looks like function-ish
    return toks[-1]

# =========================
# Mode Detection
# =========================

def is_explain_question(q: str):
    return bool(re.search(r"在做什麼|是幹嘛|what does|用途|功能|幹嘛用", q, re.IGNORECASE))

def is_sink_trace_question(q: str):
    return bool(re.search(r"誰.*call|誰.*呼叫|who calls|callers of|哪裡呼叫", q, re.IGNORECASE))

def is_control_flow_question(q: str):
    # explicitly ask about switch/case/graph/decision/branch
    return bool(re.search(r"case|switch|分支|條件|decision|graph|flow v2|led_cases", q, re.IGNORECASE))

def is_data_flow_question(q: str):
    # mask/flag/bit/assign and similar
    return bool(re.search(r"mask|flag|bit|bitmap|位元|遮罩|設定|set|清除|clear|IS_MASK|SET_MASK|CLR_MASK|assign|賦值|=", q, re.IGNORECASE))

# =========================
# Modes
# =========================

def explain_mode(question: str):
    fn = extract_function_name(question)
    if not fn:
        return "我抓不到你要看的 function 名稱（請直接打 function name，例如：cpu_fan_update 在做什麼）"

    code = get_function_code(fn)
    if not code:
        return f"找不到 function: {fn}"

    prompt = f"""
You are a firmware code analyst.

Explain this C function:

{code}

Explain:
1) Purpose (一句話)
2) Main logic steps
3) Key variables / flags
4) Whether it directly affects hardware (PWM/register/GPIO/EC RAM)
Return in Traditional Chinese.
"""
    return ask_llm(prompt)

def sink_trace_mode(question: str):
    fn = extract_function_name(question)
    if not fn:
        # fallback: last identifier
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
        fn = toks[-1] if toks else None

    if not fn:
        return "我抓不到 target function 名稱（請直接打：誰 call cpu_fan_update）"

    callers = find_callers(fn)
    if not callers:
        return f"沒有找到 call {fn} 的 function（或 Chroma 裡還沒 index 到相關檔案）"

    return f"{fn} 被以下 function 呼叫:\n" + "\n".join(callers)

def control_flow_mode(question: str):
    if not Path(STRUCTURED_JSON).exists():
        return "找不到 decision JSON（請確認 output/flow_v2/led_cases_lite.json 是否存在）"

    data = json.loads(Path(STRUCTURED_JSON).read_text(encoding="utf-8"))

    prompt = f"""
You are a firmware control-flow analyzer.

Use this structured graph:

{json.dumps(data, indent=2)}

Question:
{question}

Answer strictly based on the graph.
Return in Traditional Chinese.
"""
    return ask_llm(prompt)

def data_flow_mode(question: str):
    """
    For questions like:
      - cpu_fan_update 哪裡設定 mask
      - 這段 code 哪裡 set flag
      - 哪裡 clear bit
    We'll:
      1) load the function code
      2) ask LLM to pinpoint mask/flag operations (exact lines/expressions)
      3) optionally list callers if question implies flow entry
    """
    fn = extract_function_name(question)
    if not fn:
        return "我抓不到你要看的 function 名稱（請直接打 function name，例如：cpu_fan_update 哪裡設定 mask）"

    code = get_function_code(fn)
    if not code:
        return f"找不到 function: {fn}"

    callers_hint = ""
    if re.search(r"flow|哪裡進來|從哪裡呼叫|誰呼叫", question, re.IGNORECASE):
        callers = find_callers(fn)
        if callers:
            callers_hint = "Possible callers:\n" + "\n".join(callers)
        else:
            callers_hint = "Possible callers: (none found by heuristic search)"

    prompt = f"""
You are a firmware data-flow analyst.

Task:
- Identify where masks/flags/bits are set/cleared/checked/updated in the given function.
- Point to the exact expressions/macros, and explain what each does.
- If the code uses macros like IS_MASK_SET/IS_MASK_CLEAR/SET_MASK/CLR_MASK, explain them based on usage.
- If there is no mask operation in this function, say so clearly and suggest where to search next (e.g., callers or related setters).

Function name: {fn}

{callers_hint}

C code:
{code}

Return in Traditional Chinese with bullet points.
"""
    return ask_llm(prompt)

# =========================
# Router
# =========================

def route(q: str):
    # Priority matters:
    # data-flow questions should NOT fall into control-flow graph by default.
    if is_sink_trace_question(q):
        return sink_trace_mode(q)

    if is_data_flow_question(q):
        return data_flow_mode(q)

    if is_explain_question(q):
        return explain_mode(q)

    if is_control_flow_question(q):
        return control_flow_mode(q)

    # default: explain mode is usually safer than decision graph
    # because most questions are about real code, not the LED case graph.
    return explain_mode(q)

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-q", "--question")
    args = parser.parse_args()

    if args.interactive:
        while True:
            q = input("firmware> ").strip()
            if q.lower() in ["exit", "quit"]:
                break
            print(route(q))
    else:
        if not args.question:
            print("請提供問題")
            return
        print(route(args.question))

if __name__ == "__main__":
    main()