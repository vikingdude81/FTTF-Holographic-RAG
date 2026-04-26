# FTTF Holographic RAG: Fourier-Transform Holographic Memory for Multi-Project Retrieval-Augmented Generation

**vikingdude81**

*FTTF-Holographic-RAG v0.2.0 — April 2026*

---

## Abstract

We present **FTTF Holographic RAG**, a local retrieval-augmented generation (RAG) system that uses the Fourier Transform as its core binding mechanism for navigating large, multi-project codebases. The system addresses a practical gap: existing RAG pipelines treat all documents as a flat pool, forcing the retriever to score every document against every query. We instead compress each project into a single high-dimensional vector by associating document embeddings with random keys through element-wise multiplication in the Fourier frequency domain — a technique from the Holographic Reduced Representations (HRR) literature — enabling sub-linear project-level filtering before fine-grained file recovery.

The retrieval pipeline operates in two stages. Stage 1 ranks projects by cosine similarity between the query embedding and per-project centroid vectors in 10,000D space, achieving **57.2% Precision@1** on a 113-project, 50,000-document benchmark. Stage 2 recovers the top-k files within the highest-ranked projects using direct 768D cosine similarity, achieving **89.4% P@1**. Complementary BM25 keyword search and a tunable linear fusion mode are also provided. The system runs fully locally on commodity hardware using Ollama for embedding and ChromaDB for document storage, with no external API dependencies.

---

## 1. Introduction

Retrieval-augmented generation has become a standard approach for grounding language model responses in a specific document corpus. The dominant architecture maintains a vector database of document chunk embeddings and responds to each query by performing approximate nearest-neighbor search over all chunks. This approach scales well for single-project corpora, but degrades in two ways when the document store spans many independent projects:

1. **Semantic dilution**: A query relevant to one project retrieves semantically similar (but contextually irrelevant) passages from other projects.
2. **Flat search overhead**: Scoring every chunk against every query does not exploit the hierarchical structure of the data.

FTTF Holographic RAG addresses both problems by introducing a project-level routing layer backed by holographic memory. Each project is represented as a single compressed vector that encodes the semantic content of all its documents. Routing selects a small set of candidate projects before falling through to standard embedding-level retrieval, dramatically reducing the search space at fine-grained stages.

The system also provides a hybrid retrieval mode that linearly fuses holographic vector scores with BM25 keyword scores, handling queries that do not embed well — such as exact function names, error codes, or library identifiers.

We make the following contributions:

- A practical two-stage retrieval architecture combining holographic project routing with standard cosine file recovery.
- An empirical analysis of why direct HRR cosine similarity fails for this task (§4.2), and why centroid-based routing is preferable in production.
- A full implementation with persistence, cache integrity, hit-based promotion, and hybrid search, released under MIT license.

---

## 2. Background

### 2.1 Fourier-Domain Binding and Superposition

FTTF represents associations between documents and keys as operations in the **Fourier frequency domain**. Every embedding vector is transformed into its frequency representation via the Discrete Fourier Transform (DFT); associations are formed by multiplying spectra; and the result is transformed back to signal space via the Inverse DFT. This is the same mathematical framework used in Holographic Reduced Representations (HRR; Plate, 1995), which formalized this approach for vector-symbolic AI.

The two core operations are **binding** (associating a key with a value) and **superposition** (accumulating multiple associations into a single vector).

**Binding** is element-wise multiplication in Fourier space, followed by an inverse transform back to signal space:

$$\text{bind}(\mathbf{k}, \mathbf{v}) = \text{IFFT}\bigl(\text{FFT}(\mathbf{k}) \odot \text{FFT}(\mathbf{v})\bigr)$$

where $\odot$ denotes element-wise multiplication of complex spectra, and only the real part of the IFFT result is retained. The Fourier Transform converts each vector into a set of frequency components (magnitude and phase at each frequency). Multiplying two spectra fuses those frequency representations into a single combined signal — this is what makes the bound vector encode *both* the key and the value simultaneously, in a fixed-size output.

Note: by the convolution theorem, frequency-domain multiplication is equivalent to circular convolution in the signal domain. "Circular convolution" is the mathematical name for what this operation produces — but the operation itself is fundamentally a Fourier-domain multiplication, not the other way around.

**Superposition** accumulates $N$ bound pairs into a single memory vector by simple addition:

$$M = \sum_{i=1}^{N} \text{bind}(\mathbf{k}_i, \mathbf{v}_i) = \sum_{i=1}^{N} \text{IFFT}\bigl(\text{FFT}(\mathbf{k}_i) \odot \text{FFT}(\mathbf{v}_i)\bigr)$$

The memory $M$ is a single real-valued vector of the same dimensionality $D$ as the inputs, regardless of how many pairs $N$ have been stored.

**Unbinding** approximately recovers a stored value by dividing out the key's spectrum. Since FFT coefficients of a random unit-norm vector have approximately unit magnitude, the complex conjugate $\overline{\text{FFT}(\mathbf{k})}$ serves as an approximate frequency-domain inverse:

$$\tilde{\mathbf{v}}_j \approx \text{IFFT}\bigl(\text{FFT}(M) \odot \overline{\text{FFT}(\mathbf{k}_j)}\bigr)$$

This is approximate frequency-domain division: multiplying by the conjugate spectrum of the key approximately cancels the key's contribution to each frequency bin, leaving the value's spectrum to dominate. The recovery is imperfect because the superposition mixes all stored pairs, and dividing out one key leaves residual contributions from all others — cross-talk noise:

$$\tilde{\mathbf{v}}_j = \mathbf{v}_j + \underbrace{\sum_{i \neq j} \text{IFFT}\bigl(\text{FFT}(\mathbf{k}_i \circledast \mathbf{v}_i) \odot \overline{\text{FFT}(\mathbf{k}_j)}\bigr)}_{\text{cross-talk noise}}$$

The noise floor decreases as dimension $D$ increases (more frequency bins dilute the interference) and as the stored keys become more orthogonal to one another (less spectral overlap between keys means cleaner division).

### 2.2 BM25

Okapi BM25 (Robertson et al., 1994) is a term-frequency-based ranking function that scores document $d$ against query $q$ as:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \bigl(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\bigr)}$$

where $f(t,d)$ is the term frequency of $t$ in $d$, $|d|$ is the document length in tokens, $\text{avgdl}$ is the average document length, and:

$$\text{IDF}(t) = \log\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

with $N$ the number of documents and $n(t)$ the number of documents containing $t$. FTTF uses $k_1 = 1.5$ and $b = 0.75$.

### 2.3 Johnson-Lindenstrauss Random Projection

The Johnson-Lindenstrauss lemma states that a random linear map approximately preserves pairwise distances. FTTF uses this to project 768-dimensional embeddings into a 10,000-dimensional holographic space. The projection matrix $R \in \mathbb{R}^{768 \times 10000}$ has entries drawn i.i.d. from $\mathcal{N}(0, 1/\sqrt{d_{\text{target}}})$. For a query or document embedding $\mathbf{e} \in \mathbb{R}^{768}$, the projected vector is:

$$\mathbf{p} = \frac{R^\top \mathbf{e}}{\|R^\top \mathbf{e}\|}$$

The fixed random seed (42) ensures the same projection is applied consistently across all sessions, so stored and query vectors remain in the same space.

---

## 3. System Architecture

The system is organized as a pipeline of five stages: ingestion, embedding, cache construction, query processing, and result fusion. Figure 1 below summarizes the data flow.

```
 ┌──────────────────────────────────────────────────────────┐
 │  Ingestion                                               │
 │  Source files → chunker.py → ChromaDB (768D embeddings)  │
 └───────────────────────────┬──────────────────────────────┘
                             │
 ┌───────────────────────────▼──────────────────────────────┐
 │  Cache Construction (cache.py)                           │
 │  Per-project:                                            │
 │    768D embeddings → JL projection → 10kD embeddings     │
 │    10kD embeddings → orthogonalize keys → HRR binding    │
 │    10kD embeddings → mean → centroid vector              │
 │  Persisted: HMAC-signed pickle, mtime-invalidated        │
 └───────────────────────────┬──────────────────────────────┘
                             │
 ┌───────────────────────────▼──────────────────────────────┐
 │  Query Processing (search.py)                            │
 │  Query text → Ollama (768D) → JL projection (10kD)       │
 │                                                          │
 │  Stage 1: cosine(query_10k, centroid_10k) per project    │
 │           → hit boost → top-k projects                   │
 │  Stage 2: cosine(query_768, file_emb_768) within top-3   │
 │           → top-k files per project                      │
 └──────────────────────────────────────────────────────────┘
```

---

## 4. Technical Methods

### 4.1 Embedding and Dimension Projection

All text is embedded using `nomic-embed-text` via a local Ollama service, producing 768-dimensional unit-norm vectors. These are projected to 10,000D using the JL random projection described in §2.3. The higher dimensionality matters for two reasons specific to Fourier-domain binding:

1. **Frequency resolution and noise.** A $D$-dimensional vector has $D$ independent frequency bins. When $N$ key-value pairs are superimposed, the cross-talk noise power at any one bin is spread across all $D$ bins. The signal-to-noise ratio of the recovered value scales as $\sqrt{D/N}$ — so more frequency bins (larger $D$) means cleaner recovery from the same number of stored items.
2. **Distance preservation.** The JL projection approximately preserves cosine similarities from the original embedding space: $\cos(\mathbf{p}_i, \mathbf{p}_j) \approx \cos(\mathbf{e}_i, \mathbf{e}_j)$ with high probability for large $D$. This ensures that semantic relationships present in the 768D embedding space are maintained after projection, so Fourier-domain operations on the 10,000D vectors still reflect meaning.

The projection matrix is constructed once with a fixed seed and reused across all sessions, ensuring that cache vectors and live query vectors pass through the same transformation and remain in a consistent frequency space.

### 4.2 Why Direct Memory Cosine Fails

A critical design decision in FTTF concerns what to compare against at query time. Naively, one might compute $\cos(\mathbf{q}_{\text{proj}},\ M_P)$ — the cosine between the projected query and the superposition memory vector for project $P$. In practice, this achieves only **1.5% P@1** — barely above random chance (0.9% for 113 projects).

The reason is rooted in what Fourier-domain multiplication actually does to a vector. When we compute $\text{bind}(\mathbf{k}, \mathbf{v}) = \text{IFFT}(\text{FFT}(\mathbf{k}) \odot \text{FFT}(\mathbf{v}))$, every frequency bin of the output receives the product of the corresponding bins of $\mathbf{k}$ and $\mathbf{v}$. In particular, the **phase** of each output bin is the *sum* of the phases of $\mathbf{k}$ and $\mathbf{v}$ at that frequency. Because $\mathbf{k}$ is a random unit vector, its phases are random — so binding rotates the phase of every frequency bin of $\mathbf{v}$ by a different random amount.

The superposition $M_P$ is the sum of $N$ such phase-scrambled vectors. The result has approximately zero mean across dimensions with high variance — it points in no consistent direction in the embedding space. Taking the cosine between $M_P$ and any query embedding produces near-zero output regardless of what is stored, because there is no directional relationship between the scrambled memory and any original embedding.

**The memory is designed for retrieval via unbinding, not for direct comparison.** It answers "given the key $\mathbf{k}_j$, what value was stored with it?" — not "how similar is this memory to this query?"

**Fourier unbinding as a retrieval heuristic** achieves **37.5% P@1**. When we treat the query embedding itself as an approximate key and divide it out of the memory spectrum:

$$\tilde{\mathbf{v}} = \text{IFFT}\bigl(\text{FFT}(M_P) \odot \overline{\text{FFT}(\mathbf{q}_{\text{proj}})}\bigr)$$

This partially reverses the phase scrambling for any stored pair whose key has a similar frequency profile to the query. When the query is semantically close to a stored key, that pair's value contributes disproportionately to $\tilde{\mathbf{v}}$ because its phases are most cleanly cancelled. However, since query embeddings were never stored as keys, the cancellation is imperfect, the noise floor is high, and the signal is weak — hence the gap between 37.5% and the 57.2% achieved by the centroid approach.

### 4.3 Centroid-Based Project Routing

The **centroid** of project $P$ is:

$$\mathbf{c}_P = \frac{1}{N_P} \sum_{i=1}^{N_P} \mathbf{p}_i$$

where $\mathbf{p}_i$ are the 10,000D projected embeddings of the $N_P$ documents in project $P$.

Since projection is a linear map, $\mathbf{c}_P = \frac{1}{N_P} R^\top \sum_i \mathbf{e}_i$. The cosine between a query and the centroid is:

$$\cos(\mathbf{q}_{\text{proj}},\ \mathbf{c}_P) \propto \mathbf{q}_{\text{proj}} \cdot \mathbf{c}_P = \mathbf{q}_{\text{proj}} \cdot \frac{1}{N_P} \sum_{i=1}^{N_P} \mathbf{p}_i$$

This is the average cosine between the query and all documents in the project (up to normalization). It is a clean, interpretable signal: it is high when many documents in the project are semantically relevant to the query. Centroid routing achieves **57.2% P@1** with no tuning.

### 4.4 Key Orthogonalization

Random unit vectors in high dimensions are nearly orthogonal with high probability, but "nearly" is not "exactly." In a superposition of $N$ pairs, the cross-talk noise at retrieval for pair $j$ is:

$$\mathbf{n}_j = \sum_{i \neq j} (\mathbf{k}_i \circledast \mathbf{v}_i) \circledast \mathbf{k}_j^{-1}$$

Each term in $\mathbf{n}_j$ arises because dividing out $\mathbf{k}_j$'s spectrum does not cleanly remove the contribution of $\mathbf{k}_i \neq \mathbf{k}_j$ — the residual is proportional to the spectral overlap between keys, which is related to their dot product $|\mathbf{k}_i \cdot \mathbf{k}_j|$. By making keys more orthogonal — minimizing pairwise dot products, and therefore spectral overlap — we reduce the noise floor.

FTTF applies an iterative Gram-Schmidt-like update to the key matrix $K \in \mathbb{R}^{N \times D}$:

$$K \leftarrow K - \frac{\alpha}{D}\,(KK^\top - I) \cdot K, \qquad K \leftarrow \text{row-normalize}(K)$$

where $\alpha = 0.3$ and $I$ is omitted for the off-diagonal correction (the diagonal is zeroed in $G = KK^\top$ before multiplication). This is iterated twice. For large $N$, keys are orthogonalized in chunks of up to 200 to keep the computation tractable.

### 4.5 Multi-Bank Binding

Superposition capacity — the number of pairs that can be stored and recovered before noise overwhelms signal — grows as $O(\sqrt{D})$ for random keys. For projects with many documents, a single superposition vector becomes saturated.

FTTF distributes bindings across $B = 4$ banks using round-robin assignment:

$$M^{(b)} = \sum_{\substack{i=1 \\ i \bmod B = b}}^{N} \mathbf{k}_i \circledast \mathbf{v}_i$$

At retrieval, banks are summed before unbinding: $M = \sum_b M^{(b)}$. This is equivalent to a single superposition, but the intermediate representation enables future extensions such as per-bank promotion or selective retrieval.

For large projects ($N > 60$), a **hierarchical** scheme is applied: documents are processed in chunks of 25, each chunk's bindings are summed into a sub-vector, and sub-vectors are assigned to banks round-robin. This bounds memory usage and avoids building a single superposition over hundreds of items.

### 4.6 Signal Processing Decode Pipeline

When HRR unbinding is used (e.g., for theoretical analysis), the raw unbind output is processed through a three-step decode pipeline:

**Step 1 — Normalize:**
$$\mathbf{u} = \frac{\tilde{\mathbf{v}}}{\|\tilde{\mathbf{v}}\|}$$

**Step 2 — Sharpen** ($p = 0.7$, applied element-wise):
$$\text{sharpen}(x;\ p) = \text{sign}(x) \cdot (|x| + \epsilon)^p$$

With $p < 1$, this is a *compressive* nonlinearity: smaller-magnitude components are amplified relative to larger ones. This partially counteracts the effect of cross-talk noise, which tends to inflate a few dominant dimensions.

**Step 3 — CORVACS-lite** (Controlled Rectification and Value Amplitude Compression/Saturation; $a = 1.5$):
$$\text{corvacs}(x;\ a) = \frac{\tanh(a \cdot x)}{a}$$

This applies soft saturation: no single dimension can dominate the output vector. The scale factor $1/a$ maintains approximate magnitude. Together, sharpening and CORVACS-lite improve the signal-to-noise ratio of the recovered vector before cosine similarity is computed.

### 4.7 Two-Stage Search

The production retrieval pipeline does not use HRR unbinding for ranking. It uses:

**Stage 1 — Project ranking (10,000D centroid cosine):**

$$s_P = \cos(\mathbf{q}_{\text{proj}},\ \mathbf{c}_P) + 0.1 \cdot \frac{h(P)}{\max_j h(j)}$$

where $h(P)$ is the hit count for project $P$ from the HitTracker (§4.8), and the second term is a small promotion boost (capped at 1.0 total). The top-$k$ projects are returned.

**Stage 2 — File ranking (768D cosine, within top-3 projects):**

$$s_f = \cos(\mathbf{q}_{\text{emb}},\ \mathbf{e}_f)$$

where $\mathbf{q}_{\text{emb}}$ is the raw 768D query embedding and $\mathbf{e}_f$ is the stored 768D embedding of file $f$. This is applied only within the top-3 ranked projects to bound computation. File results are the top-$k_f$ per project.

This two-stage design exploits the different properties of the two representations: 10,000D centroids provide fast, noise-robust project routing; 768D cosine provides precise file-level recovery within a narrowed candidate set.

### 4.8 Hit-Based Promotion

The system tracks how many times each project has appeared in search results, both total hits and distinct session counts. At query time, frequently recalled projects receive a small additive boost (up to +0.1) proportional to their normalized hit frequency:

$$s_P^{\text{boosted}} = \min\!\left(s_P + 0.1 \cdot \frac{h(P)}{\max_j h(j)},\ 1.0\right)$$

This implements a lightweight relevance feedback loop: projects that are repeatedly accessed across sessions are inferred to be broadly relevant to the user's work context, and are surfaced more readily. Projects exceeding a session-count threshold ($\tau = 3$ by default) are flagged as "promotable" in search results.

Hit data is persisted to JSON and updated after each query via atomic file replacement.

### 4.9 BM25 Keyword Search

The keyword index uses Okapi BM25 as described in §2.2. The tokenizer applies several preprocessing steps relevant to code corpora:

1. **CamelCase splitting**: `handleWebSocketAuth` → `handle Web Socket Auth` (regex boundary insertion before uppercase transitions)
2. **Lowercasing and alphanumeric extraction**
3. **Stop-word removal**: A minimal set of English function words; common code identifiers (e.g., `return`, `def`, `class`) are intentionally preserved

This tokenization means a query for `websocket` will match documents containing `WebSocketHandler`, and a query for `connection refused` will match both prose descriptions and error message strings.

BM25 scores are normalized to $[0, 1]$ by dividing by the maximum score in the result set before returning.

### 4.10 Hybrid Score Fusion

The hybrid search mode fuses normalized vector scores and normalized BM25 scores via a linear combination:

$$s_{\text{hybrid}}(q, d) = \alpha \cdot \hat{s}_{\text{vec}}(q, d) + (1 - \alpha) \cdot \hat{s}_{\text{kw}}(q, d)$$

where $\hat{s}_{\text{vec}}$ are min-max normalized file-level cosine scores (normalized across all projects), $\hat{s}_{\text{kw}}$ are normalized BM25 scores, and $\alpha \in [0, 1]$ is a user-configurable blending parameter (default 0.5).

The union of document sets from both retrievers is scored: documents absent from one retriever receive a score of 0.0 from that retriever. This means a document that scores highly on only one mode can still appear in results, while documents that score well on both modes are strongly promoted.

### 4.11 Cache Integrity and Invalidation

The holographic cache (project vectors, centroids, file embeddings, keys) is serialized with Python's `pickle` and protected by an HMAC-SHA256 signature:

$$\text{sig} = \text{HMAC-SHA256}(k_{\text{machine}},\ \text{pickle}(\text{cache}))$$

The signing key $k_{\text{machine}}$ is derived from the absolute cache path and the OS username, then hashed with SHA-256. This key is machine-local: a cache file cannot be verified on a different machine or by a different user, preventing subtle corruption from cross-environment sharing.

Cache validity is verified against a content fingerprint computed from ChromaDB document metadata:

$$\text{fingerprint} = \text{SHA256}\!\left(\text{count} \,\|\, \bigsqcup_i \text{mtime}(d_i)\right)$$

where $\bigsqcup$ denotes sorted concatenation. If the fingerprint changes (new documents ingested, documents modified), the cache is transparently rebuilt on the next search request.

---

## 5. Implementation

### 5.1 Module Organization

| Module | Responsibility |
|--------|---------------|
| `engine.py` | HRR primitives, signal processing, dimension projection, hit tracking |
| `cache.py` | Cache construction, HMAC persistence, Ollama health, mtime fingerprinting |
| `search.py` | Two-stage search API, hybrid fusion, BM25 orchestration |
| `keyword.py` | BM25 index, tokenizer |
| `ingest.py` | Directory crawling, deduplication, ChromaDB upsert |
| `chunker.py` | Paragraph-aware text chunking with overlap |
| `config.py` | `HoloConfig` dataclass with all tuneable parameters |
| `cli.py` | `fttf-holo` CLI entry point with 7 subcommands |

### 5.2 Dependencies

The core runtime requires only NumPy, ChromaDB, and the `requests` library. Ollama is an external service dependency, not a Python package. PDF extraction requires PyMuPDF, which is an optional dependency. There are no ties to any specific LLM framework.

### 5.3 Persistence Layer

All on-disk I/O uses atomic rename (`os.replace`) to prevent partial writes from corrupting the cache. The holographic cache is a single signed pickle file. Hit data is stored as plain JSON. All paths are configurable via `HoloConfig`.

---

## 6. Evaluation

All benchmarks were run against a collection of **113 software projects** containing **50,000 documents** ingested into ChromaDB, embedded with `nomic-embed-text`.

### 6.1 Project Routing (Stage 1)

| Strategy | P@1 |
|----------|-----|
| Centroid 10,000D cosine | **57.2%** |
| HRR unbinding + decode | 37.5% |
| HRR direct cosine similarity | 1.5% |
| Random baseline | 0.9% |

The centroid strategy outperforms HRR unbinding by 19.7 percentage points. HRR direct cosine is marginally above random, confirming the theoretical argument in §4.2.

### 6.2 File Recovery (Stage 2)

| Strategy | P@1 |
|----------|-----|
| 768D cosine within top-3 projects | **89.4%** |
| HRR unbinding (file level) | 38% |

Direct 768D cosine similarity within the project scope achieves near-ceiling performance. The large gap confirms that the holographic representation is most valuable at the routing layer, where it compresses many documents into a single comparable vector, rather than at the file-recovery layer, where standard cosine similarity on the original embedding space is optimal.

### 6.3 Discussion

**Why does centroid work so well for routing?**
The centroid is a sufficient statistic for project relevance when documents are approximately uniformly sampled from the project's topic space. A query about "websocket authentication" will have high cosine with embeddings of files that handle websocket auth, and the centroid will reflect this if those files are well-represented in the project. The 10,000D JL projection approximately preserves these cosine similarities while providing a common dimensionality for the holographic layer.

**When does centroid fail?**
For very large projects with diverse sub-modules, the centroid may be diluted toward a generic "this project does many things" direction. Targeted sub-project routing (using scoped search via `search_scoped`) is the recommended workaround.

**Why retain HRR at all?**
The HRR binding layer provides several benefits that centroids alone do not:
- **Theoretical framework**: The superposition vector is a compact encoding that supports unbinding-based access — useful for future work on key-guided retrieval.
- **Interpolation**: The HRR vector and the centroid vector occupy the same 10,000D space and can be blended.
- **Non-uniform weighting**: Unlike the centroid (which weights all documents equally), the multi-bank superposition can be modified to weight documents by importance (e.g., hit frequency) by scaling values before binding.

---

## 7. Limitations and Future Work

**Centroid dilution at scale.** For projects with thousands of documents spanning unrelated sub-topics, the centroid loses discriminative power. Hierarchical project decomposition (sub-project routing) is a natural extension.

**Static projection matrix.** The JL projection matrix is fixed at construction time. If the embedding model changes or is fine-tuned, the projection must be regenerated and the cache rebuilt. A learnable projection could adapt to the embedding space.

**In-memory index.** The BM25 index and file embedding maps are loaded entirely into memory. For very large corpora (>100K documents), an on-disk inverted index and approximate nearest-neighbor structures (e.g., HNSW) would be needed.

**HRR key storage.** The current implementation stores all binding keys on disk for potential future use. For large projects this adds significant cache size. A seeded key generator would allow keys to be reconstructed without storage.

**Approximate P@1 measurement.** The benchmark was run against a specific corpus and query set; formal reproducibility requires a public benchmark release.

---

## 8. Conclusion

FTTF Holographic RAG demonstrates that Holographic Reduced Representations are practically useful for the *routing* stage of multi-project retrieval, even though direct HRR cosine similarity is unsuitable for ranking. By combining a centroid-based project router in a high-dimensional holographic space with standard 768D cosine file recovery, the system achieves 57.2% P@1 project routing and 89.4% P@1 file recovery on a 113-project benchmark, operating fully locally with no external API dependencies.

The key practical insight is that HRR binding is not a replacement for cosine similarity — it is a complementary operation that compresses a variable-size document set into a fixed-size vector amenable to associative retrieval. When paired with appropriate signal processing and orthogonalized keys, this compression is semantically meaningful enough for coarse routing, while fine-grained recovery is delegated to the well-understood cosine similarity in the original embedding space.

---

## References

- Plate, T. A. (1995). *Holographic reduced representations*. IEEE Transactions on Neural Networks, 6(3), 623–641.
- Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M. M., & Gatford, M. (1994). *Okapi at TREC-3*. NIST Special Publication 500-225.
- Smolensky, P. (1990). *Tensor product variable binding and the representation of symbolic structures in connectionist systems*. Artificial Intelligence, 46(1–2), 159–216.
- Johnson, W. B., & Lindenstrauss, J. (1984). *Extensions of Lipschitz mappings into a Hilbert space*. Contemporary Mathematics, 26, 189–206.
- Gayler, R. W. (2004). *Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience*. arXiv:cs/0412059.
- Lewis, P., et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks*. NeurIPS 2020.

---

*Source code: [https://github.com/vikingdude81/FTTF-Holographic-RAG](https://github.com/vikingdude81/FTTF-Holographic-RAG)*
*License: MIT*
