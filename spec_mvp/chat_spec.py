import json
import re
import requests
import chromadb
from typing import Any, Dict, List, Tuple, Optional

# =========================
# CONFIG
# =========================
CHROMA_PATH = "output/spec_semantic_db"
COLLECTION = "spec_sections"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# 你現在的模型設定（沿用你檔案裡的）
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "mxbai-embed-large:latest"

TOP_N = 50
TOP_PER_FILE = 3

# cross-file 用比較嚴；file-level（指定檔案）放寬/不砍
CROSS_FILE_DIST_TH = 0.82
FILE_LEVEL_DIST_TH = 0.95  # 也可以設成 None 表示不過濾

# embeddings endpoint 長度限制保護（避免 500）
EMBED_MAX_CHARS = 1100
EMBED_SHRINK_STEPS = [1100, 900, 700, 500, 350, 250]

# 回答時引用原文片段長度（避免 prompt 太大）
DOC_SNIP_CHARS = 900

# =========================
# OLLAMA
# =========================
def ollama_chat(system_prompt: str, user_prompt: str) -> str:
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ollama_embed(text: str) -> List[float]:
    """
    Ollama /api/embeddings 有時會對 prompt 長度/Token 有硬限制，超過會 500。
    這裡做：
      - 先裁到安全長度（預設 1100 chars）
      - 如果還 500，逐步縮短重試
    """
    def _call(prompt: str) -> List[float]:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": prompt},
            timeout=180,
        )
        r.raise_for_status()
        emb = r.json().get("embedding")
        if not emb:
            raise RuntimeError("Empty embedding from Ollama embeddings API")
        return emb

    prompt = (text or "").strip()
    if len(prompt) > EMBED_MAX_CHARS:
        prompt = prompt[:EMBED_MAX_CHARS]

    last_err: Optional[Exception] = None
    for n in EMBED_SHRINK_STEPS:
        try:
            p = prompt[:n] if len(prompt) > n else prompt
            return _call(p)
        except requests.HTTPError as e:
            last_err = e
            # 只針對 500 做縮短重試；其他錯誤直接丟出
            if e.response is None or e.response.status_code != 500:
                raise
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else RuntimeError("Embedding failed after shrinking")


# =========================
# UTIL
# =========================
def extract_pdf_filename(text: str) -> str:
    """
    從問題中抽出 .pdf 檔名（含空白），若沒有則回傳空字串
    """
    if not text:
        return ""
    m = re.search(r"([A-Za-z0-9_\-\. ]+\.pdf)", text, flags=re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def safe_json_extract(s: str) -> Optional[dict]:
    if not s:
        return None
    try:
        start = s.find("{")
        end = s.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        return json.loads(s[start:end])
    except Exception:
        return None


def meta_label(meta: Dict[str, Any]) -> str:
    sec = (meta.get("section") or "").strip()
    page = meta.get("page_start")
    if sec:
        return f"章節 {sec}"
    if page is not None and str(page).strip() != "":
        return f"頁碼 {page}"
    return "未知位置"


def is_changelog(summary: str) -> bool:
    s = (summary or "").lower()
    bad = ["revision", "history", "changelog", "版本歷史", "修改", "變更", "update", "modified"]
    return any(b in s for b in bad)


def _json_list_to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join([str(v) for v in x])
    if isinstance(x, str):
        # keywords/signals 可能是 JSON string
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return " ".join([str(v) for v in obj])
            return str(obj)
        except Exception:
            return x
    return str(x)


def build_blob_for_filter(meta: Dict[str, Any], doc_text: str = "") -> str:
    """
    must_terms 後過濾用的 blob：把 summary/keywords/signals/content_preview/doc snippet 都納入
    """
    parts = []
    parts.append(meta.get("summary") or "")
    parts.append(_json_list_to_text(meta.get("keywords")))
    parts.append(_json_list_to_text(meta.get("signals")))
    parts.append(meta.get("content_preview") or "")

    if doc_text:
        parts.append(doc_text[:DOC_SNIP_CHARS])

    return " ".join(parts).lower()


def pass_must_terms(meta: Dict[str, Any], must_terms: List[str], doc_text: str = "") -> bool:
    if not must_terms:
        return True
    blob = build_blob_for_filter(meta, doc_text=doc_text)
    return any(t.lower() in blob for t in must_terms)


def compact_doc_snip(doc: str) -> str:
    """
    從 documents 抽一段人類可讀 snippet（避免太長）
    """
    if not doc:
        return ""
    # 優先保留 [DOC] 區塊內容；如果沒有，就直接截
    # 你的 build index 版本通常會有 [DOC]... [EMBED_TEXT]
    if "[DOC]" in doc and "[EMBED_TEXT]" in doc:
        core = doc.split("[EMBED_TEXT]")[0]
    else:
        core = doc
    core = core.strip()
    return core[:DOC_SNIP_CHARS]


# =========================
# QUERY PLANNER (NO HARDCODE TOPIC)
# =========================
def rewrite_query(question: str, recent_context: str = "") -> Dict[str, Any]:
    """
    用 LLM 把自然語言問題改寫成更適合 spec semantic retrieval 的查詢 query，
    同時吐 must_terms（後過濾用）和 file_hint（如果有提 .pdf）。
    """
    file_hint_from_text = extract_pdf_filename(question)

    system = "You are a query planner for firmware specification retrieval. Return STRICT JSON only."
    user = f"""
請把使用者問題改寫成「適合用向量檢索 spec 章節」的查詢語句。

規則：
- 只輸出 JSON（不要多任何字）。
- search_query：用英文為主（可混中英關鍵字），偏向 spec 常見語彙：requirement/specification/behavior/policy/bitfield/register/table/protocol/ACPI/SMBus/I2C/EC/shutdown/error handling... 等。
- must_terms：列出 1~5 個最關鍵詞（縮寫、method name、register like 0x39/39h、_BST、RelativeStateOfCharge、bit7...），用於後過濾。
- file_hint：如果問題裡有 .pdf 檔名就填；沒有就空字串。

最近對話（可用來理解「那個呢」這種承接）：
{recent_context}

使用者問題：
{question}

請輸出 JSON：
{{
  "search_query": "...",
  "must_terms": ["..."],
  "file_hint": ""
}}
"""
    out = ollama_chat(system, user).strip()
    data = safe_json_extract(out) or {}

    # 保底欄位
    sq = (data.get("search_query") or question).strip()
    mt = data.get("must_terms")
    if not isinstance(mt, list):
        mt = []
    fh = (data.get("file_hint") or "").strip()

    # 若 LLM 沒抓到檔名，但使用者句子有，就補上
    if not fh and file_hint_from_text:
        fh = file_hint_from_text

    # 防呆：must_terms 最多 5 個
    mt = [str(x).strip() for x in mt if str(x).strip()]
    mt = mt[:5]

    return {"search_query": sq, "must_terms": mt, "file_hint": fh}


# =========================
# SEMANTIC SEARCH
# =========================
def semantic_search(
    query: str,
    file_filter: str = "",
    n_results: int = TOP_N,
) -> List[Tuple[Dict[str, Any], float, str]]:
    """
    回傳 (meta, dist, doc_snip)
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)

    q_emb = ollama_embed(query)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["metadatas", "distances", "documents"],  # ✅ 加 documents
    )

    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]

    items: List[Tuple[Dict[str, Any], float, str]] = []
    for m, d, doc in zip(metas, dists, docs):
        meta = m or {}
        dist = float(d) if d is not None else 9999.0

        if is_changelog(meta.get("summary", "")):
            continue

        if file_filter:
            if file_filter.lower() not in (meta.get("file", "") or "").lower():
                continue

        doc_snip = compact_doc_snip(doc or "")
        items.append((meta, dist, doc_snip))

    items.sort(key=lambda x: x[1])
    return items


def format_hits_for_llm(
    items: List[Tuple[Dict[str, Any], float, str]],
    per_file: int = TOP_PER_FILE,
) -> str:
    """
    把命中整理成 LLM 可讀的 context（含原文片段）
    """
    # group by file
    file_map: Dict[str, List[Tuple[Dict[str, Any], float, str]]] = {}
    for meta, dist, snip in items:
        f = meta.get("file", "UNKNOWN")
        file_map.setdefault(f, []).append((meta, dist, snip))

    context_lines: List[str] = []
    for f in sorted(file_map.keys()):
        top_hits = sorted(file_map[f], key=lambda x: x[1])[:per_file]
        context_lines.append(f"\n文件: {f}")
        for meta, dist, snip in top_hits:
            context_lines.append(
                f"  - {meta_label(meta)} (dist={dist:.3f})\n"
                f"    summary: {meta.get('summary','')}\n"
                f"    preview: {meta.get('content_preview','')}\n"
                f"    snippet:\n{snip}\n"
            )
    return "\n".join(context_lines).strip()


# =========================
# ANSWER BUILDERS
# =========================
def answer_cross_file(question: str, recent_context: str) -> str:
    plan = rewrite_query(question, recent_context)
    q = plan["search_query"]
    must_terms = plan["must_terms"]

    items = semantic_search(q, file_filter="")

    # cross-file 做距離過濾（避免飄太遠）
    items = [(m, d, snip) for (m, d, snip) in items if d < CROSS_FILE_DIST_TH]
    # must_terms 後過濾（提升精準度）
    items = [(m, d, snip) for (m, d, snip) in items if pass_must_terms(m, must_terms, doc_text=snip)]

    if not items:
        return "⚠ 找不到相關規範。（可能 query 太短或 spec 沒有該主題）"

    context = format_hits_for_llm(items, per_file=TOP_PER_FILE)

    system_prompt = (
        "你是Firmware Spec分析助理。"
        "請用中文回答。"
        "要『直接回答使用者問題』，並列出最關鍵的Spec文件與章節位置。"
        "如果只是相關但不是主要規範，請明確說『僅相關』。"
        "回答要依據提供的 snippet/summary，不要憑空猜。"
    )

    user_prompt = f"""
使用者問題：
{question}

檢索到的章節摘要 + 原文片段（已過濾）：
{context}

請輸出：
1) 最主要的 SPEC（檔名）
2) 對應章節/頁碼（2~3 個就好）
3) 每個章節一句話說明它在規範什麼
"""
    return ollama_chat(system_prompt, user_prompt)


def answer_file_level(question: str, recent_context: str, file_hint: str) -> str:
    plan = rewrite_query(question, recent_context)
    q = plan["search_query"]
    must_terms = plan["must_terms"]

    # 指定檔案：強制 file_filter
    if not file_hint:
        file_hint = plan.get("file_hint", "")

    if not file_hint:
        return "⚠ 你看起來是要查特定檔案，但我沒抓到 .pdf 檔名。"

    items = semantic_search(q, file_filter=file_hint)

    # file-level：放寬 threshold（甚至可以不過濾）
    if FILE_LEVEL_DIST_TH is not None:
        items = [(m, d, snip) for (m, d, snip) in items if d < FILE_LEVEL_DIST_TH]

    # must_terms 後過濾（若過濾後為空，就退回不過濾，避免『找不到』）
    filtered = [(m, d, snip) for (m, d, snip) in items if pass_must_terms(m, must_terms, doc_text=snip)]
    if filtered:
        items = filtered

    if not items:
        return f"⚠ 在 {file_hint} 找不到相關內容。（可能這份PDF章節summary太短或主題措辭不同）"

    # 直接取前幾個（同一檔案）
    top = items[: min(6, len(items))]
    context = format_hits_for_llm(top, per_file=6)

    system_prompt = (
        "你是Firmware Spec助理。"
        "請用中文回答，並明確指出章節/頁碼。"
        "如果章節號不存在，請用頁碼定位。"
        "回答要依據提供的 snippet/summary，不要憑空猜。"
    )
    user_prompt = f"""
問題：
{question}

在指定文件內檢索到的章節摘要 + 原文片段：
{context}

請輸出：
- 這份文件跟問題最相關的規範點是什麼（條列 3 點內）
- 每點要附上章節/頁碼定位
"""
    return ollama_chat(system_prompt, user_prompt)


def answer_section_lookup(section_hint: str) -> str:
    """
    章節精準查：用 section_hint 直接比對 metadata.section
    注意：有些PDF不會有章節號，這個模式就不一定命中。
    """
    if not section_hint:
        return "⚠ 沒提供章節號。"

    # 用 section 當 query 先拿一批，再精準比對
    query = f"section {section_hint} specification requirement"
    items = semantic_search(query, file_filter="")

    for meta, dist, snip in items:
        if (meta.get("section") or "").strip() == str(section_hint).strip():
            return (
                f"[{meta.get('file','UNKNOWN')} - 章節 {section_hint}] (dist={dist:.3f})\n"
                f"summary: {meta.get('summary','')}\n"
                f"preview: {meta.get('content_preview','')}\n"
                f"snippet:\n{snip}\n"
            )
    return "⚠ 找不到該章節（可能該PDF沒有章節號，或 parser 沒抽到）。"


# =========================
# MAIN LOOP
# =========================
def main():
    print("=== Spec Chat Mode v7 (Planner + Semantic Retrieval + Snippets) ===")
    print("輸入 exit 離開；輸入 state 查看最近上下文\n")

    conversation_memory: List[str] = []

    while True:
        question = input("firmware> ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        if question.lower() == "state":
            print("\n--- recent context ---")
            print("\n".join(conversation_memory[-5:]))
            print("----------------------\n")
            continue

        # memory
        conversation_memory.append(question)
        recent_context = "\n".join(conversation_memory[-3:])

        # 強制：如果使用者句子有 .pdf，就走 file-level
        pdf_name = extract_pdf_filename(question)
        if pdf_name:
            print(answer_file_level(question, recent_context, pdf_name))
            continue

        # 章節模式（簡單 heuristics）
        # e.g. "4.1.1 在幹嘛" / "section 4.1.1"
        m_sec = re.search(r"\b(\d+(?:\.\d+){1,5})\b", question)
        if m_sec and any(k in question.lower() for k in ["章節", "section", "在幹嘛", "做什麼", "說什麼", "內容"]):
            sec = m_sec.group(1)
            # 若句子很像「4.1.1 在幹嘛」就走 section lookup
            if len(question) <= 18:
                print(answer_section_lookup(sec))
                continue

        # 預設：cross-file（哪些 spec / 哪裡規範）
        print(answer_cross_file(question, recent_context))


if __name__ == "__main__":
    main()