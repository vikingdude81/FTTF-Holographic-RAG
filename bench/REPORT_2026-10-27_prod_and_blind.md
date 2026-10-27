# Retrieval Bench — Round 2: blind queries + production-scale multi-project bench (2026-10-27)

Follow-up to `REPORT_2026-10-27.md` (archive single-project bench). Two new benches plus a
full audit of the production RAG stack, and a production-crash library fix.

---

## 1. Production stack audit (AI_Command_Center launcher)

Startup folder (`shell:startup`) contains: `Ollama.lnk` (tray app, so Ollama serves at
logon), `start_rag_service.bat`, `start_qwen_service.bat`, `start_agent_service.bat`,
`start_lgi_dashboard.lnk`, `HarmonicFieldAI.lnk`.

- **RAG service** = `fttf_holo_rag` (production import resolves to this repo) over
  Chroma collection `my_codebase`: **55,951 docs / 217 projects**, holographic cache
  `holo_cache.pkl` 4.7 GB, built **2026-04-16**.
- **Interference check (Ollama):** `api/ps` empty — nothing resident. `qwen3.5:122b`
  (81 GB Q4_K_M MoE, 262K ctx) loads only on demand; if loaded it spills to system RAM
  (slower), but embedding runs are unaffected when it isn't. `nomic-embed-text` 274 MB.
- **Staleness (headline finding):** Chroma itself is frozen at Apr 16. No project created
  after June exists anywhere in the retrieval layer: theology-philosophy-archive,
  motion-lgi, FALCON, continuant-probe absent; `lpf_pipeline` still 12 old files.
  brain.py has been blind to ~6 months of work.
- **Boot-crash bug (fixed today):** `mtime_hash()` and `build()` issued unbatched
  `coll.get(limit=60_000)`; at 55k docs that raises `too many SQL variables`.
  mtime_hash swallowed it and fell back to a bare count ("55951") ≠ the stored sha256
  → every launch decided the cache was stale → rebuild crashed on its own unbatched get.
  The service could not boot. Patched `_get_all()` batching + `mtime_hash`; verified the
  patched hash reproduces the stored `9b0b936c…` exactly, so the next launch loads
  cleanly and staleness detection is real again. 117/117 tests pass.
- **Hygiene (unfixed):** single PDFs registered as their own "projects"
  (`Consciousnetics.pdf` and its duplicate `Consciousnetics (1).pdf`, `ML.pdf`,
  `deeplearningforfinance*.pdf`); clone dupes (`crypto-ml…` + `-main`,
  `FTTF-fournier (1)/(2)`). These become yoked attractors in routing.
- **Security (unfixed):** `start_agent_service.bat` contains a plaintext OpenAI key
  [REDACTED] — move to `.env`/user-env and rotate.
- `holo_hits.json` frozen Apr 25 — hit promotion has been dormant since April.

## 2. Production bench — real multi-project addressing

16 blind queries (written from corpus knowledge, gold projects/files from cache file
lists; yoked near-dupes listed as acceptable gold sets), unscored staleness probe
`stale-newton` (files created after Apr cache). 217 projects / 55,951 docs.
Fresh hits file → zero promotion boost. Holo path replicates production search exactly
(stage-1 10kD centroid cosine, stage-2 cached 768D within top-3 projects).

| system | route@1 | route@3 | filehit@10 | median lat |
|---|---|---|---|---|
| holo (2-stage) | **3/16** | 9/16 | 1/9 | 2,100 ms |
| dense (flat cosine, all docs) | 8/16 | 12/16 | 1/9 | 2,087 ms |
| bm25 (flat) | 7/16 | 13/16 | 6/9 | **157 ms** |
| **hybrid (vec+BM25)** | 8/16 | **14/16** | **6/9** | 2,468 ms |

**Verdict: FTTF's headline claim (holographic project addressing) does not hold at
217-project scale on this corpus.** Centroid routing loses to flat cosine head-to-head.
Failure taxonomy observed per-query:
1. **Centroid dilution** — big heterogeneous projects (harmonic-field-consciousness, 893
   files) have diluted centroids; single-document "projects" (Consciousnetics.pdf) have
   pure-signal centroids and win domain queries. Bigger ≠ better addressed.
2. **Yoked attractors** — duplicate clones (PDF copies, `-main` dupes) split the signal
   and burn routing slots (FTTF-fournier (1)+(2) both ranking).
3. **Adjacent-domain capture** — cet-epi → QUANTUM-GENETICS ("evolution"),
   auxrng/qrng-batt → entropic_bridge. Semantically close, strictly wrong project.
4. **Stage-2 fragility** — file ranking only sees top-3 projects' cached embeddings;
   a routing error is unrecoverable. 1,500-char snippet embedding also under-ranks gold
   files vs BM25 (1/9 vs 6/9).
5. **Staleness probe** — all systems miss `regime_discovery.py` (post-Apr); BM25 returns
   old `newton_basin_map.py` cousins. Invisible-work penalty is real, not hypothetical.

Speed framing: holographic routing has no cost advantage at this scale — 217×10kD cosine
and 55k×768D cosine are both sub-3 s. Its addressing model only pays off at far larger
project counts, and it currently *costs* recall to get there.

## 3. Archive blind round (round 2 control)

12 paraphrased user-question queries (NOT link-graph-derived, not title wording) +
2 negative probes. Same corpus/store as round 1 (101 docs).

| system | recall@5 | recall@10 | MRR@10 | round-1 MRR |
|---|---|---|---|---|
| fttf_search | 0.917 | 1.000 | 0.741 | 0.687 |
| fttf_keyword | 0.917 | 0.917 | 0.792 | 0.637 |
| **fttf_hybrid** | **1.000** | **1.000** | **0.958** | 0.816 |
| dense_nomic | 1.000 | 1.000 | 0.850 | 0.626 |
| grep_naive | 1.000 | 1.000 | 0.674 | 0.589 |

No gold-set circularity: rankings hold and tighten under blind queries (hybrid 0.816→
0.958). Only single miss: fttf_search on blind-pyrrhon. Negatives: every system returns
confident nearest-neighbor garbage for k8s/sourdough queries — **no abstention threshold
exists at any layer**; retrieval can't say "the answer is not in the corpus".

## 4. Recommendations

1. Use **hybrid flat (vector+BM25)** as the production default today; it won both benches.
   If latency matters, BM25-first with vector re-rank only on the top ~200 (BM25 was
   13x faster at equal recall on files).
2. Fix corpus hygiene before any holo rerun: dedup PDF clones, fold `-main` dupes,
   never register single files as projects, exclude `.git/node_modules/build`.
3. Re-ingest + rebuild the holo cache **as an explicit overnight job** (≥55k docs ≈
   hours via Ollama). Do not rely on boot-time rebuild.
4. Add an abstention gate (score floor / margin test) — measured need.
5. Holographic addressing: keep as research line, don't gate production retrieval on it;
   revisit only after corpus hygiene + larger project counts, with centroid sub-clustering
   for >300-file projects (fixes dilution failure mode 1).

Files: `bench/prod_gold.json`, `bench/prod_bench.py`, `bench/prod_results.json`,
`bench/gold_blind.json`, `bench/results_blind_2026-10-27.json`, library fix in
`fttf_holo_rag/cache.py` (`_get_all`, batched `mtime_hash`).
