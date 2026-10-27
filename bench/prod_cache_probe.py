"""List production cache projects with per-project file counts."""
from fttf_holo_rag.cache import load_signed_cache

CACHE = r"C:\Users\akbon\Projects\AI_Command_Center\chroma_db_storage\holo_cache.pkl"
data = load_signed_cache(CACHE)
print("doc_count:", data["doc_count"], "version:", data["version"], "mtime_hash:", data["mtime_hash"][:16])
files = data["files"]
tot = sum(len(v) for v in files.values())
print("cached projects:", len(files), "cached files total:", tot)
for name in sorted(files, key=lambda k: -len(files[k])):
    print(f"{len(files[name]):6d}  {name}")
