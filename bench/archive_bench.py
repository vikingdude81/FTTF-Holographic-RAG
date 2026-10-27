"""archive-retrieval-bench: FTTF vs standard retrieval on the theology-philosophy archive.

Systems: fttf_search (2-stage holo), fttf_keyword (BM25), fttf_hybrid (fusion),
dense (raw nomic-embed cosine, doc-level), grep (naive term-frequency, ripgrep-style).

Gold: bench/gold.json (link-graph-derived). Metrics: recall@5 (any gold),
recall@10 (all gold), MRR@10. Run from repo root with the bench venv.
"""
import json, os, re, sys, math
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = r"C:\Users\akbon\OneDrive\Documents\GitHub\theology-philosophy-archive"
STORE = os.path.join(ROOT, "bench_store")

from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.search import HoloRAG
import chromadb
import requests

gold_doc = json.load(open(os.path.join(ROOT, "bench", "gold.json"), encoding="utf-8"))
QUERIES = gold_doc["queries"]

cfg = HoloConfig(chroma_path=STORE, chroma_collection="archive_bench",
                 holo_cache_path=os.path.join(STORE, "holo_cache.pkl"),
                 holo_hits_path=os.path.join(STORE, "holo_hits.json"))
rag = HoloRAG(cfg)

col = chromadb.PersistentClient(path=STORE).get_collection("archive_bench")
docs = col.get(include=["embeddings", "metadatas"])
ids = docs["ids"]
emb = {i: e for i, e in zip(ids, docs["embeddings"])}
texts = {}  # filepath -> full on-disk text (for dense + grep, beyond ingest truncation)
for i in ids:
    try:
        texts[i] = open(i, encoding="utf-8", errors="ignore").read()
    except OSError:
        texts[i] = ""

STOP = set("the a an of in on to and or for is are was were be been it its that this with as at by from what which who whom how why not no do does did you your we our they their i me my he she his her him us this these those there here then than so if but can would could should will shall may might must about into out up down over under more most other some any all none one two".split())

def toks(s):
    return [w for w in re.findall(r"[a-z][a-z0-9_\-']+", s.lower()) if w not in STOP]

def ollama_embed(text):
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={"model": "nomic-embed-text", "prompt": text[:3800]}, timeout=120)
    j = r.json()
    if "embedding" not in j:
        raise RuntimeError(f"embed failed: {j}")
    return j["embedding"]

# cache dense doc embeddings (full text, first 6000 chars)
DENSE_PATH = os.path.join(ROOT, "bench", "dense_embeddings.json")
if os.path.exists(DENSE_PATH):
    dense = {k: v for k, v in json.load(open(DENSE_PATH)).items()}
else:
    dense = {}
    for i in ids:
        if i not in dense:
            dense[i] = ollama_embed(texts[i][:6000])
    json.dump(dense, open(DENSE_PATH, "w"))

def norm(fp):
    return os.path.basename(fp.replace("\\", "/").replace("/", os.sep))

def cos(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(x*x for x in b))
    return num / (da*db) if da and db else 0.0

def sys_fttf_search(q):
    res = rag.search(q, top_k_projects=1, top_k_files=10)
    out = []
    for pr in res.projects:
        for f in pr.top_files:
            out.append(f.filepath)
    return out

def sys_keyword(q):
    return [r.filepath for r in rag.keyword_search(q, top_k=10).results]

def sys_hybrid(q):
    return [r.filepath for r in rag.hybrid_search(q, top_k=10).results]

def sys_dense(q):
    qe = ollama_embed(q)
    return sorted(ids, key=lambda i: -cos(qe, dense[i]))[:10]

def sys_grep(q):
    qt = Counter(toks(q))
    n_docs = len(ids)
    df = Counter()
    per_doc = {}
    for i in ids:
        tf = Counter(toks(texts[i]))
        per_doc[i] = tf
        for t in set(tf):
            df[t] += 1
    def score(i):
        s = 0.0
        for t, c in qt.items():
            if t in per_doc[i]:
                s += c * per_doc[i][t] * (math.log(n_docs / max(df[t], 1)))
        return s
    return sorted(ids, key=lambda i: -score(i))[:10]

SYSTEMS = {"fttf_search": sys_fttf_search, "fttf_keyword": sys_keyword,
           "fttf_hybrid": sys_hybrid, "dense_nomic": sys_dense, "grep_naive": sys_grep}

results = {}
for name, fn in SYSTEMS.items():
    rec5 = rec10 = 0
    mrr_sum = 0.0
    per_q = {}
    for Q in QUERIES:
        ranked = [norm(p) for p in fn(Q["query"])]
        gold = [norm("x/" + g) if "/" not in g else norm(g) for g in Q["gold"]]
        top5, top10 = ranked[:5], ranked[:10]
        hit5 = any(g in top5 for g in gold)
        hit10 = any(g in top10 for g in gold)
        rr = 0.0
        for r, d in enumerate(top10, 1):
            if d in gold:
                rr = 1.0 / r
                break
        rec5 += hit5; rec10 += hit10; mrr_sum += rr
        per_q[Q["id"]] = {"top5": top5[:5], "hit5": hit5, "rank_best": (next((r for r, d in enumerate(top10, 1) if d in gold), None))}
    n = len(QUERIES)
    results[name] = {"recall@5": round(rec5/n, 3), "recall@10": round(rec10/n, 3),
                     "MRR@10": round(mrr_sum/n, 3), "per_query": per_q}
    print(f"{name:14s} recall@5={rec5/n:.3f}  recall@10={rec10/n:.3f}  MRR@10={mrr_sum/n:.3f}")

out = {"date": "2026-10-27", "n_queries": len(QUERIES), "corpus_docs": len(ids), "systems": results}
json.dump(out, open(os.path.join(ROOT, "bench", "results_2026-10-27.json"), "w"), indent=1)
print("\nwrote bench/results_2026-10-27.json")
