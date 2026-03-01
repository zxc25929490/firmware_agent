import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import requests
import chromadb

# =========================
# CONFIG
# =========================
CHROMA_PATH = "output/spec_db"
COLLECTION = "spec"
OUTPUT_JSON = "spec_mvp/sections_db.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b-instruct"

# 避免爆 token：但不要太短，不然只剩頁首頁尾
MAX_CONTENT_LENGTH = 9000

# 用於存 debug 的 head/tail
HEAD_CHARS = 1200
TAIL_CHARS = 1200

# 如果章節清理後太短，標記 low_quality
MIN_CLEAN_CHARS = 350

# =========================
# CLEANING (降低「頁眉/目錄/雜訊」帶來的幻覺)
# =========================
def normalize_ws(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def looks_like_toc(line: str) -> bool:
    """
    目錄/索引常見型態：
      3.1.4 Something ......... 64
    """
    l = line.strip()
    if not l:
        return False
    if re.search(r"\.{5,}\s*\d+\s*$", l):
        return True
    return False


def looks_like_page_footer_header(line: str) -> bool:
    l = line.strip()
    if not l:
        return False

    # 常見頁碼/機密字樣/版權頁腳
    footer_patterns = [
        r"^\d+\s*/\s*\d+\s*$",
        r"^page\s*\d+\s*(of\s*\d+)?$",
        r"^confidential$",
        r"^copyright",
        r"^asus\b",
        r"^revision\b",
        r"^version\b",
    ]
    for p in footer_patterns:
        if re.search(p, l, flags=re.IGNORECASE):
            return True

    # 只有一兩個字 + 重複符號線
    if re.fullmatch(r"[-_=]{4,}", l):
        return True

    return False


def clean_section_text(text: str) -> Tuple[str, List[str]]:
    """
    回傳 (clean_text, flags)
    flags 會標記：toc_like / too_short / mostly_noise
    """
    flags = []
    text = normalize_ws(text)
    if not text:
        return "", ["empty"]

    lines = [ln.strip() for ln in text.split("\n")]
    kept = []
    toc_hits = 0
    noise_hits = 0

    for ln in lines:
        if not ln:
            continue
        if looks_like_toc(ln):
            toc_hits += 1
            continue
        if looks_like_page_footer_header(ln):
            noise_hits += 1
            continue
        # 太短的行通常是頁眉殘渣（但也可能是標題；保守起見：>=4字才留）
        if len(ln) < 4:
            noise_hits += 1
            continue
        kept.append(ln)

    clean = "\n".join(kept).strip()

    # flags
    if toc_hits >= 2:
        flags.append("toc_like")
    if len(clean) < MIN_CLEAN_CHARS:
        flags.append("too_short")
    if (noise_hits + toc_hits) > max(10, len(lines) * 0.6):
        flags.append("mostly_noise")

    return clean, flags


def safe_json_parse(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()

    # 有些模型會把 JSON 包在字串或多一些雜訊
    # 嘗試抓第一個 {...} 區塊
    try:
        if raw.startswith("{") and raw.endswith("}"):
            return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    return {}


# =========================
# LLM 產生「可驗證」 summary + keywords + evidence
# =========================
def generate_summary_keywords_evidence(file_name: str, section: str, page: Any, clean_content: str) -> Dict[str, Any]:
    """
    目標：低幻覺、可追溯
    - 每個 key point 都要附 evidence（從原文摘錄，<= 30 字）
    - 不允許說原文沒有的具體名詞（例如 BIOS memory training）除非 evidence 有
    """
    prompt = f"""
你是一個 firmware 規格分析助手。你必須「只根據原文」產生結果，不能自行補完猜測。

請根據以下章節原文，輸出 STRICT JSON（不要多任何字）：

規則：
- summary.what：用繁體中文 1~2 句話說「本章在規範什麼」
- summary.key_points：3~6 點，每點 1 句，描述 requirement/behavior/interface/flow/bitfield/table 等
- summary.evidence：對應 key_points 的證據片段（從原文逐字摘錄），每點 10~30 字，總字數不超過 180 字
- keywords：5~12 個「英文技術關鍵字」，必須能在原文中找到或非常通用（例如 EC, ACPI, GPIO, LED, HID, SMBus）
- 如果原文資訊不足或太像目錄/標題頁，請在 quality_flags 加上 "low_confidence"

輸出 JSON schema：
{{
  "summary": {{
    "what": "",
    "key_points": ["", ""],
    "evidence": ["", ""]
  }},
  "keywords": ["", ""],
  "quality_flags": []
}}

檔案：{file_name}
章節：{section}
page_start：{page}

原文：
{clean_content}
"""

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )
    r.raise_for_status()

    raw = r.json().get("response", "")
    parsed = safe_json_parse(raw)

    # 保底欄位
    summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
    what = (summary.get("what") or "").strip()
    key_points = summary.get("key_points") if isinstance(summary.get("key_points"), list) else []
    evidence = summary.get("evidence") if isinstance(summary.get("evidence"), list) else []
    keywords = parsed.get("keywords") if isinstance(parsed.get("keywords"), list) else []
    qf = parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else []

    # 長度與一致性修正
    key_points = [str(x).strip() for x in key_points if str(x).strip()][:6]
    evidence = [str(x).strip() for x in evidence if str(x).strip()][:6]
    keywords = [str(x).strip() for x in keywords if str(x).strip()][:12]
    qf = [str(x).strip() for x in qf if str(x).strip()]

    # 若 evidence 數量不足，補空字串對齊
    while len(evidence) < len(key_points):
        evidence.append("")

    return {
        "summary": {
            "what": what,
            "key_points": key_points,
            "evidence": evidence,
        },
        "keywords": keywords,
        "quality_flags": qf,
    }


# =========================
# MAIN
# =========================
def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)

    data = col.get()
    docs = data["documents"]
    metas = data["metadatas"]

    # 依 file + section + page 聚合
    grouped = defaultdict(list)

    for doc, meta in zip(docs, metas):
        file_name = meta.get("source_file", "UNKNOWN")
        section = meta.get("section", "UNKNOWN")
        page = meta.get("page_start", "")

        key = (file_name, section, page)
        grouped[key].append(doc)

    sections_db: List[Dict[str, Any]] = []

    print(f"Total sections detected: {len(grouped)}")

    for (file_name, section, page), contents in grouped.items():
        full_content = "\n".join(contents)
        full_content = normalize_ws(full_content)

        # 長度裁切，但保留 tail（有些 spec table 在後段）
        if len(full_content) > MAX_CONTENT_LENGTH:
            # 取 head + tail（避免只剩開頭）
            head = full_content[: int(MAX_CONTENT_LENGTH * 0.65)]
            tail = full_content[-int(MAX_CONTENT_LENGTH * 0.35) :]
            full_content = (head + "\n...\n" + tail).strip()

        clean_content, clean_flags = clean_section_text(full_content)

        print(f"Processing: {file_name} - Section {section} (page_start={page}) flags={clean_flags}")

        # 預設輸出（就算 LLM 失敗也不會崩）
        summary_obj = {"what": "", "key_points": [], "evidence": []}
        keywords: List[str] = []
        quality_flags: List[str] = list(clean_flags)

        # 如果清理後內容太短/太雜，直接標記 low_confidence，仍可嘗試產生但會提醒
        if "too_short" in clean_flags or "mostly_noise" in clean_flags:
            quality_flags.append("low_confidence")

        try:
            result = generate_summary_keywords_evidence(
                file_name=file_name,
                section=section,
                page=page,
                clean_content=clean_content if clean_content else full_content,
            )
            summary_obj = result.get("summary", summary_obj)
            keywords = result.get("keywords", keywords)
            quality_flags = list(set(quality_flags + result.get("quality_flags", [])))
        except Exception as e:
            print("⚠ LLM error:", e)
            quality_flags = list(set(quality_flags + ["llm_failed", "low_confidence"]))

        # 存 debug head/tail（方便你看「餵了什麼」）
        debug_head = (clean_content or full_content)[:HEAD_CHARS]
        debug_tail = (clean_content or full_content)[-TAIL_CHARS:] if (clean_content or full_content) else ""

        sections_db.append(
            {
                "file": file_name,
                "section": section,
                "page_start": page,

                # 新版：summary 不是一段話，是可驗證的結構
                "summary": summary_obj.get("what", ""),
                "key_points": summary_obj.get("key_points", []),
                "evidence": summary_obj.get("evidence", []),

                "keywords": keywords,

                # 新增：品質標記
                "quality_flags": sorted(list(set(quality_flags))),

                # 原文（存詳細）
                "content": clean_content if clean_content else full_content,

                # debug（避免你以為是 3.1.4 但其實餵的是目錄頁）
                "content_head": debug_head,
                "content_tail": debug_tail,
            }
        )

    Path(OUTPUT_JSON).write_text(
        json.dumps(sections_db, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✅ sections_db.json generated")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()