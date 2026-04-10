# FTTF Holographic RAG

**Fourier Transform holographic memory for local retrieval-augmented generation.**

A standalone, framework-agnostic holographic search system that uses:
- **Stage 1**: Centroid-based project ranking (10,000D holographic vectors)
- **Stage 2**: Raw 768D cosine similarity for file-level recovery

Built on Holographic Reduced Representations (HRR) with signal processing upgrades:
orthogonalized keys, multi-bank binding, CORVACS-lite saturation, and hit-based promotion.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Your Application                   │
│  (AI Command Center / LangChain / custom pipeline)   │
├─────────────────────────────────────────────────────┤
│                  fttf_holo_rag                       │
│  HoloRAG.search()  →  Project ranking (centroids)   │
│                    →  File recovery (768D cosine)    │
├─────────────────────────────────────────────────────┤
│  engine.py         │  cache.py        │  search.py   │
│  HRR primitives    │  HMAC-signed     │  2-stage     │
│  Signal processing │  Pickle cache    │  search API  │
│  Key orthogonal.   │  Ollama embed    │  Hit tracker │
├─────────────────────────────────────────────────────┤
│           ChromaDB            │       Ollama          │
│      (document store)         │  (embedding service)  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install -e .
```

### As a Python library

```python
from fttf_holo_rag import HoloRAG, HoloConfig

# Point to your ChromaDB and Ollama
config = HoloConfig(
    chroma_path="./chroma_db_storage",
    chroma_collection="my_codebase",
    ollama_url="http://localhost:11434/api/embeddings",
)

rag = HoloRAG(config)

# Search across all projects
result = rag.search("websocket authentication")
print(result.to_text())

# Search within a specific project
files = rag.search_scoped("database connection", "my-project")
for f in files:
    print(f"  {f.relevance:.1%}  {f.filepath}")

# List all projects
projects = rag.list_projects()

# Force cache rebuild
rag.build_cache(force=True)
```

### CLI

```bash
# Search
fttf-holo search "holographic memory fourier transform"
fttf-holo search --project NanoGPT "attention mechanism"
fttf-holo search --json "websocket server"

# Build/rebuild cache
fttf-holo build
fttf-holo build --force

# List projects
fttf-holo projects

# Custom paths
fttf-holo --chroma-path ./my_db --ollama-url http://gpu:11434/api/embeddings search "query"
```

### JSON output

```bash
fttf-holo search --json "machine learning"
```
```json
{
  "query": "machine learning",
  "stage": "centroid",
  "projects": [
    {
      "name": "NanoGPT",
      "similarity": 0.63,
      "file_count": 182,
      "hits": 0,
      "top_files": [
        {"filepath": "E:\\NanoGPT\\train.py", "relevance": 0.71}
      ]
    }
  ]
}
```

## Plugin Integration

### With AI Command Center

```python
# In AI_Command_Center/tools/memory.py
from fttf_holo_rag import HoloRAG, HoloConfig

_rag = HoloRAG(HoloConfig(
    chroma_path=config.CHROMA_PATH,
    ollama_url=config.OLLAMA_URL,
))

@register_tool('search_holographic')
class SearchHolographic(BaseTool):
    def call(self, params, **kwargs):
        result = _rag.search(params['search_query'])
        return result.to_text()
```

### With LangChain

```python
from langchain.tools import tool
from fttf_holo_rag import HoloRAG

rag = HoloRAG()

@tool
def holographic_search(query: str) -> str:
    """Search codebase using holographic memory."""
    return rag.search(query).to_text()
```

## How It Works

### Stage 1: Project Identification (Centroid Ranking)

1. Documents are embedded via Ollama (nomic-embed-text, 768D)
2. Embeddings are projected to 10,000D via Johnson-Lindenstrauss
3. Per-project **centroids** (mean of 10kD embeddings) are computed
4. Query is embedded, projected, and compared via cosine similarity
5. Top projects are ranked — ~57% P@1 on 113-project benchmark

### Stage 2: File Recovery (768D Cosine)

1. Within top-3 projects, raw 768D embeddings are compared to query
2. Direct cosine similarity — **89% P@1** (vs 38% for HRR unbinding)
3. HRR is kept for theoretical research; raw cosine handles production search

### Why Not Pure HRR?

HRR's `bind()` randomizes phase via circular convolution — by design.
This makes `cos(query, superposition) ≈ 0` regardless of content.
HRR is an **item recovery** mechanism (unbind with a known key), not a similarity search tool.

**Benchmark results (50K docs, 113 projects)**:
| Strategy | P@1 |
|----------|------|
| Centroid 10kD | 57.2% |
| Raw cosine 768D (file) | 89.4% |
| HRR similarity | 1.5% |
| HRR unbinding | 37.5% |

## Configuration

All configuration via `HoloConfig` dataclass — see [config.py](fttf_holo_rag/config.py).

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `holo_dim` | 10,000 | Holographic vector dimension |
| `holo_n_banks` | 4 | Multi-bank count |
| `holo_orth_iters` | 2 | Key orthogonalization passes |
| `holo_sharpen_p` | 0.7 | Signal sharpening exponent |
| `holo_corvacs_a` | 1.5 | CORVACS-lite saturation |
| `embed_model` | nomic-embed-text | Ollama embedding model |

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.ai/) running locally with an embedding model
- ChromaDB collection with ingested documents

## License

MIT
