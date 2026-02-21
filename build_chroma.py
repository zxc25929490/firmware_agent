import os
import re
import chromadb
from chromadb.utils import embedding_functions

SOURCE_DIR = "codebase"  # 改成你的code資料夾
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "codebase"

def extract_functions_from_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    pattern = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_ \*\n]+?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*?\)\s*\{',
        re.MULTILINE
    )

    matches = pattern.finditer(content)
    functions = []

    for match in matches:
        start = match.start()
        name = match.group(2)

        brace_count = 0
        i = start
        while i < len(content):
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
            i += 1

        func_code = content[start:end]
        functions.append((name, func_code))

    return functions


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_func = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                path = os.path.join(root, file)
                print(f"Processing {path}")
                functions = extract_functions_from_file(path)

                for name, code in functions:
                    collection.add(
                        documents=[code],
                        metadatas=[{"function": name, "file": path}],
                        ids=[f"{path}:{name}"]
                    )

    print("✅ Chroma build complete.")


if __name__ == "__main__":
    main()