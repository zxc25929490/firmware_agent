import json
import re
import requests
import chromadb

# =========================
# CONFIG
# =========================
CHROMA_PATH = "output/spec_semantic_db"
COLLECTION = "spec_sections"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "mxbai-embed-large:latest"

TOP_N = 50
TOP_PER_FILE = 5


# =========================
# EMBEDDING
# =========================
def ollama_embed(text: str) -> list[float]:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=180
    )
    r.raise_for_status()
    emb = r.json().get("embedding")
    if not emb:
        raise RuntimeError("Empty embedding from Ollama")
    return emb


# =========================
# FILTER RULES
# =========================
def is_changelog(meta: dict) -> bool:
    s = (meta.get("summary") or "").lower()

    bad = [
        "版本歷史", "revision", "history", "changelog",
        "modify", "modified", "update", "變更", "修改", "新增"
    ]

    return any(b in s for b in bad)


def is_led_spec_like(meta: dict) -> bool:
    s = (meta.get("summary") or "").lower()
    kw = (meta.get("keywords") or "")

    # keywords 可能是 JSON string
    try:
        kw_list = json.loads(kw)
        kw = " ".join(kw_list).lower()
    except:
        kw = str(kw).lower()

    blob = s + " " + kw

    strong_terms = [
        "led",
        "charge led",
        "full led",
        "pwr_led",
        "chg_led",
        "backlight",
        "keyboard",
        "rgb",
        "pwm",
        "indicator"
    ]

    return any(term in blob for term in strong_terms)


# =========================
# MAIN
# =========================
def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: python spec_mvp/query_semantic.py "LED specification requirements"')
        return

    user_q = sys.argv[1].strip()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)

    # 強化查詢語句
    semantic_query = f"{user_q} behavior requirement specification control policy must shall table"

    q_emb = ollama_embed(semantic_query)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_N,
        include=["metadatas", "distances"]
    )

    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # =========================
    # FILTER
    # =========================
    items = []

    for m, d in zip(metas, dists):
        if is_changelog(m):
            continue

        if not is_led_spec_like(m):
            continue

        items.append((m, d))

    if not items:
        print("⚠ No relevant LED specification sections found.")
        return

    # =========================
    # GROUP BY FILE
    # =========================
    file_map = {}

    for m, d in items:
        f = m.get("file", "UNKNOWN")
        sec = m.get("section", "UNKNOWN")
        summ = m.get("summary", "")
        file_map.setdefault(f, []).append((sec, d, summ))

    print("\n=== Semantic LED Specification Results ===\n")

    for f in sorted(file_map.keys()):
        sorted_items = sorted(file_map[f], key=lambda x: x[1])  # distance ascending
        top_items = sorted_items[:TOP_PER_FILE]

        sec_list = ", ".join([s for s, _, _ in top_items])
        print(f"- {f} → sections: {sec_list}")

        # 顯示前兩個摘要
        for s, d, summ in top_items[:2]:
            clean_summ = (summ or "").replace("\n", " ").strip()
            print(f"    * {s} (dist={d:.4f}) {clean_summ[:140]}")

        print("")


if __name__ == "__main__":
    main()