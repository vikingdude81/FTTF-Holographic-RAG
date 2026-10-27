"""prod-bench: FTTF holographic project addressing on the REAL production corpus
(217 projects / 55,951 docs, warm cache from 2026-04-16).

Reuses the exact production code path (HoloCache pickle + DimensionProjector(seed=42)
+ engine.get_embedding) but loads the cache directly instead of HoloRAG.search(),
because HoloRAG.build() would mtime-invalidate (E:\\lpf_pipeline changed since April)
and rebuild ~55k docs. A fresh hits file means zero promotion boost, like a cold session.

Systems:
  holo      = stage1 centroid routing (10kD) + stage2 cached-768D file ranking (top-3 proj)
  dense     = flat cosine over all 55,951 Chroma embeddings
  bm25      = library KeywordIndex over Chroma snippets
  hybrid    = library hybrid_search (vector+BM25 fusion)

Metrics: routing@1/@3 (project in top-k), file-hit@10 (basename suffix match), latency.
"""
import json, os, sys, time, logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = r"C:\Users\akbon\Projects\AI_Command_Center\chroma_db_storage"
CACHE = os.path.join(PROD, "holo_cache.pkl")
HITS_TMP = os.path.join(ROOT, "bench", "prod_hits_tmp.json")

from fttf_holo_rag.cache import load_signed_cache
from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.engine import DimensionProjector, get_embedding
from fttf_holo_rag.search import HoloRAG
import chromadb

GOLD = json.load(open(os.path.join(ROOT, "bench", "prod_gold.json"), encoding="utf-8"))["queries"]
# unscored staleness probe: files created AFTER the April cache build
EXTRA = [{"id": "stale-newton", "query": "newton basin attractor repeller regime discovery gravity model",
          "gold_projects": ["lpf_pipeline"], "gold_files": ["regime_discovery.py", "regime_filter.py"], "scored": False}]

cfg = HoloConfig(chroma_path=PROD, chroma_collection="my_codebase",
                 holo_cache_path=CACHE, holo_hits_path=HITS_TMP)

print("loading signed holo cache ...", flush=True)
t0 = time.time()
data = load_signed_cache(CACHE)
cent = data["centroids"]; files = data["files"]; fileemb = data["embeddings"] or {}
proj_names = list(cent.keys())
C = np.stack([np.asarray(cent[p], dtype=np.float32) for p in proj_names])
C /= np.linalg.norm(C, axis=1, keepdims=True)
print(f"cache: {len(proj_names)} projects in {time.time()-t0:.1f}s", flush=True)

projector = DimensionProjector(cfg.embed_dim, cfg.holo_dim, seed=42)

print("loading chroma collection (flat baselines) ...", flush=True)
t0 = time.time()
col = chromadb.PersistentClient(path=PROD).get_collection("my_codebase")
ids, metas, arrs = [], [], []
off = 0
while True:
    got = col.get(limit=2000, offset=off, include=["embeddings", "metadatas"])
    if not got["ids"]:
        break
    ids += got["ids"]; metas += got["metadatas"]
    arrs.append(np.asarray(got["embeddings"], dtype=np.float32))
    off += 2000
E = np.vstack(arrs)
E /= np.linalg.norm(E, axis=1, keepdims=True)
proj_of = {i: (m or {}).get("project_name", "?") for i, m in zip(ids, metas)}
print(f"chroma: {len(ids)} docs in {time.time()-t0:.1f}s", flush=True)

rag = HoloRAG(cfg)  # keyword/hybrid side; cache not built, search path untouched

def qemb(q):
    return get_embedding(q, cfg.ollama_url, cfg.embed_model)

def sys_holo(q):
    e = np.asarray(qemb(q), dtype=np.float32)
    e10k = np.asarray(projector.project(e.tolist()), dtype=np.float32)
    e10k /= (np.linalg.norm(e10k) + 1e-12)
    sims = C @ e10k
    order = np.argsort(-sims)
    ranked_proj = [proj_names[i] for i in order[:5]]
    en = e / (np.linalg.norm(e) + 1e-12)
    ranked_files = []
    for p in ranked_proj[:3]:
        emb_map = fileemb.get(p) or {}
        scored = []
        for fp, fe in emb_map.items():
            fv = np.asarray(fe, dtype=np.float32)
            n = np.linalg.norm(fv)
            if n < 1e-12: continue
            scored.append((float(fv @ en) / n, fp))
        scored.sort(reverse=True)
        ranked_files += [fp for _, fp in scored[:10]]
    return ranked_proj, ranked_files[:10]

def sys_dense(q):
    en = np.asarray(qemb(q), dtype=np.float32)
    en /= (np.linalg.norm(en) + 1e-12)
    s = E @ en
    order = np.argsort(-s)[:10]
    ranked_ids = [ids[i] for i in order]
    return list(dict.fromkeys([proj_of[i] for i in ranked_ids]))[:3], ranked_ids

def sys_keyword(q):
    res = rag.keyword_search(q, top_k=10)
    fp = [r.filepath for r in res.results]
    return list(dict.fromkeys([r.project for r in res.results]))[:3], fp

def sys_hybrid(q):
    res = rag.hybrid_search(q, top_k=10)
    fp = [r.filepath for r in res.results]
    return list(dict.fromkeys([r.project for r in res.results]))[:3], fp

SYSTEMS = {"holo": sys_holo, "dense": sys_dense, "bm25": sys_keyword, "hybrid": sys_hybrid}

def base(p): return os.path.basename(str(p).replace("\\", "/"))

results = {s: {"r1": 0, "r3": 0, "fh": 0, "nfh": 0, "lat": []} for s in SYSTEMS}
detail = []
for Q in GOLD + EXTRA:
    q = Q["query"]
    gp = set(Q["gold_projects"]); gf = set(Q["gold_files"])
    row = {"id": Q["id"], "scored": Q.get("scored", True)}
    for sname, fn in SYSTEMS.items():
        t0 = time.time()
        rp, rf = fn(q)
        dt = time.time() - t0
        results[sname]["lat"].append(dt)
        r1 = 1 if (rp and rp[0] in gp) else 0
        r3 = 1 if any(p in gp for p in rp[:3]) else 0
        row[sname] = {"top_proj": rp[:3], "route1": r1, "route3": r3,
                      "top_files": [base(f) for f in rf[:5]]}
        if Q.get("scored", True):
            results[sname]["r1"] += r1; results[sname]["r3"] += r3
            if gf:
                results[sname]["nfh"] += 1
                if any(any(base(f).endswith(g) for g in gf) for f in rf):
                    results[sname]["fh"] += 1
    detail.append(row)
    print(f"[{Q['id']}] done", flush=True)

N = sum(1 for Q in GOLD if Q.get("scored", True))
NF = results["holo"]["nfh"]
print(f"\n=== PROD BENCH: {N} scored queries over {len(proj_names)} projects / {len(ids)} docs ===")
print(f"{'system':8} | route@1 | route@3 | filehit@10 | median lat")
for s in SYSTEMS:
    r = results[s]
    lat = np.median(r["lat"]) * 1000
    fh = f"{r['fh']}/{NF}" if NF else "n/a"
    print(f"{s:8} | {r['r1']}/{N}   | {r['r3']}/{N}   | {fh:10} | {lat:.0f} ms")

out = {"summary": {s: {"route1": results[s]["r1"], "route3": results[s]["r3"],
                       "filehit": results[s]["fh"], "nfileq": results[s]["nfh"],
                       "median_ms": round(np.median(results[s]["lat"]) * 1000)} for s in SYSTEMS},
       "n_docs": len(ids), "n_projects": len(proj_names), "queries": detail}
json.dump(out, open(os.path.join(ROOT, "bench", "prod_results.json"), "w"), indent=1)
print("saved bench/prod_results.json")
