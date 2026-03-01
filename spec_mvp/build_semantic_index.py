import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import requests
import chromadb

# =========================
# CONFIG
# =========================
SECTIONS_DB = "spec_mvp/sections_db.json"

CHROMA_PATH = "output/spec_semantic_db"
COLLECTION = "spec_sections"

# Ollama embeddings endpoint
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

# embedding model
EMBED_MODEL = "mxbai-embed-large:latest"

BATCH_SIZE = 32

# =========================
# "存得越詳細越好" 的存儲策略
# =========================
# Chroma documents 內存原文（每 chunk 上限）
MAX_STORE_CHARS = 20000

# metadata 內存 preview（讓你現有 chat_spec.py 只 include metadatas 也看得到內容）
MAX_META_PREVIEW_CHARS = 1200

# 章節太長切 chunk（推薦開啟，否則一個段落太大 embedding 也不穩）
ENABLE_CHUNKING = True
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 600

# =========================
# embeddings endpoint 長度限制（你說大約 1000）
# =========================
# 我們把送去 /embeddings 的 prompt 控制在保守範圍
EMBED_MAX_CHARS = 1100
# 若仍遇到 500，就縮短重試
EMBED_SHRINK_STEPS = [1100, 900, 700, 500, 350, 250]

# =========================
# 可檢索訊號抽取（提升命中「threshold / bit / table」類）
# =========================
EXTRACT_SIGNALS = True
MAX_SIGNAL_ITEMS = 80


# =========================
# UTIL
# =========================
def stable_id(file_name: str, section: str, page_start: Any, chunk_idx: int) -> str:
    raw = f"{file_name}||{section}||{page_start}||chunk={chunk_idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_ws(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return [""]

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_signals(text: str) -> List[str]:
    """
    把 spec 裡常見可檢索信號抽出來：
    - 0x?? / ??h
    - bit fields
    - ACPI methods (_BST/_BIF/_Qxx...)
    - HID/descriptor/report
    - VID/PID
    - threshold/warning/critical/shutdown/SOC
    """
    if not text:
        return []

    sig = set()

    # Hex / register-ish
    for m in re.findall(r"\b0x[0-9a-fA-F]{1,8}\b", text):
        sig.add(m.lower())
    for m in re.findall(r"\b[0-9A-Fa-f]{2,4}h\b", text):
        sig.add(m.lower())

    # bit fields
    for m in re.findall(r"\bbit\s*\d{1,2}\b", text, flags=re.IGNORECASE):
        sig.add(m.lower().replace(" ", ""))

    # ACPI method like _BST, _BIF, _Qxx
    for m in re.findall(r"\b_[A-Z0-9]{2,4}\b", text):
        sig.add(m)

    # HID descriptor / report / usage
    for m in re.findall(
        r"\b(Report\s*ID\s*\d+|Usage\s*Page|Usage\s*\([^)]+\)|HID\s*Descriptor|Report\s*Descriptor)\b",
        text,
        flags=re.IGNORECASE,
    ):
        sig.add(re.sub(r"\s+", " ", m).strip())

    # VID/PID patterns
    for m in re.findall(
        r"\b(VID|PID)\s*[:=]?\s*(0x[0-9a-fA-F]{4}|[0-9a-fA-F]{4}h|\d{4,5})\b",
        text,
        flags=re.IGNORECASE,
    ):
        sig.add(f"{m[0].upper()}={m[1]}")

    # threshold hints
    for m in re.findall(
        r"\b(threshold|warning level|critical level|shutdown|low battery|soc|state of charge)\b",
        text,
        flags=re.IGNORECASE,
    ):
        sig.add(m.lower())

    out = list(sig)
    out.sort()
    return out[:MAX_SIGNAL_ITEMS]


# =========================
# OLLAMA EMBEDDINGS (安全長度 + 500 自動縮短重試)
# =========================
def ollama_embed(text: str) -> List[float]:
    def _call(prompt: str) -> List[float]:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": prompt},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"Ollama embeddings empty response: {data}")
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
            # 其他非 HTTP error 也做一次縮短嘗試，避免偶發
            continue

    raise last_err if last_err else RuntimeError("Embedding failed after shrinking retries")


# =========================
# EMBED TEXT (短、高資訊密度，避免超長)
# =========================
def build_embed_text(sec: Dict[str, Any], content_chunk: str, chunk_idx: int, chunk_total: int) -> str:
    file_ = sec.get("file", "") or ""
    section = str(sec.get("section", "") or "")
    page_start = str(sec.get("page_start", "") or "")

    summary = (sec.get("summary", "") or "").strip()
    keywords = sec.get("keywords", []) or []
    if not isinstance(keywords, list):
        keywords = []
    kw_str = ", ".join([str(k) for k in keywords[:30]])

    content_chunk = normalize_ws(content_chunk)
    content_snip = content_chunk[:500]  # 只放小段，避免 embeddings endpoint 爆長

    signals = []
    if EXTRACT_SIGNALS:
        signals = extract_signals(content_chunk + "\n" + summary + "\n" + kw_str)
    sig_str = ", ".join(signals[:MAX_SIGNAL_ITEMS])

    text = f"""
FILE: {file_}
SECTION: {section}
PAGE_START: {page_start}
CHUNK: {chunk_idx + 1}/{max(1, chunk_total)}

SUMMARY: {summary}

KEYWORDS: {kw_str}

SIGNALS: {sig_str}

CONTENT_SNIP: {content_snip}
""".strip()

    return text


# =========================
# MAIN
# =========================
def main():
    sections_path = Path(SECTIONS_DB)
    if not sections_path.exists():
        print(f"❌ Missing {SECTIONS_DB}. Please run build_sections.py first.")
        return

    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    print(f"Loaded sections: {len(sections)}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 重新建立 collection（存在就拿）
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION)

    # 先取得已有 ids，避免重複 add
    existing = set()
    try:
        got = col.get(include=["ids"])
        existing = set(got.get("ids", []))
    except Exception:
        existing = set()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    added = 0
    skipped = 0
    failed = 0

    for i, sec in enumerate(sections, start=1):
        file_name = sec.get("file", "UNKNOWN") or "UNKNOWN"
        section = str(sec.get("section", "UNKNOWN"))
        page_start = sec.get("page_start", "")

        summary = sec.get("summary", "") or ""
        keywords = sec.get("keywords", []) or []
        if not isinstance(keywords, list):
            keywords = []

        content = normalize_ws(sec.get("content", "") or "")

        # chunking（章節太長切 chunk）
        if ENABLE_CHUNKING:
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks = [content]

        chunk_total = len(chunks)

        for chunk_idx, content_chunk in enumerate(chunks):
            _id = stable_id(file_name, section, page_start, chunk_idx)
            if _id in existing:
                skipped += 1
                continue

            # embedding 用：短 prompt（內含 signals）
            embed_text = build_embed_text(sec, content_chunk, chunk_idx, chunk_total)

            # 存得詳細：documents 存「原文 chunk」(可長) + 也附上 embed_text（方便 debug）
            stored_content = (content_chunk or "")[:MAX_STORE_CHARS]
            content_preview = stored_content[:MAX_META_PREVIEW_CHARS]

            if EXTRACT_SIGNALS:
                sigs = extract_signals(stored_content + "\n" + summary)
            else:
                sigs = []

            try:
                emb = ollama_embed(embed_text)
            except Exception as e:
                failed += 1
                print(
                    f"⚠ Embed failed: {file_name} {section} page {page_start} "
                    f"chunk {chunk_idx + 1}/{chunk_total} -> {e}"
                )
                continue

            # documents：原文 + embed_text（便於回查）
            doc_text = f"""[DOC]
FILE: {file_name}
SECTION: {section}
PAGE_START: {page_start}
CHUNK: {chunk_idx + 1}/{chunk_total}

{stored_content}

[EMBED_TEXT]
{embed_text}
""".strip()

            ids.append(_id)
            docs.append(doc_text)
            metas.append(
                {
                    "file": file_name,
                    "section": section,
                    "page_start": str(page_start),
                    "chunk_idx": int(chunk_idx),
                    "chunk_total": int(chunk_total),
                    "summary": summary,
                    "keywords": json.dumps(keywords, ensure_ascii=False),

                    # 讓你 chat_spec.py 不改 include=["documents"] 也能看到內容
                    "content_preview": content_preview,

                    # 強化檢索後過濾
                    "signals": json.dumps(sigs, ensure_ascii=False),
                }
            )
            embs.append(emb)

            if len(ids) >= BATCH_SIZE:
                col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                added += len(ids)
                ids, docs, metas, embs = [], [], [], []

        if i % 100 == 0:
            print(f"Progress {i}/{len(sections)} ... added={added} skipped={skipped} failed={failed}")

    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        added += len(ids)

    print(f"✅ Done. Added={added}, skipped(existing)={skipped}, failed(embed)={failed}")
    print(f"DB path: {CHROMA_PATH}, collection: {COLLECTION}")


if __name__ == "__main__":
    main()