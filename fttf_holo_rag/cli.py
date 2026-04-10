"""
CLI entry point for FTTF Holographic RAG.

Usage:
    fttf-holo search "websocket authentication"
    fttf-holo build --force
    fttf-holo projects
    fttf-holo search --project myapp "database connection"
    fttf-holo ingest ./my-project ./other-code
    fttf-holo prep ./research-papers ./chunked-output --include-pdf
"""

import argparse
import json
import logging
import sys

from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.search import HoloRAG


def _load_config(args) -> HoloConfig:
    """Build config from CLI args, falling back to defaults."""
    kwargs = {}
    if args.chroma_path:
        kwargs['chroma_path'] = args.chroma_path
    if args.collection:
        kwargs['chroma_collection'] = args.collection
    if args.ollama_url:
        kwargs['ollama_url'] = args.ollama_url
    if args.cache_path:
        kwargs['holo_cache_path'] = args.cache_path
    if args.dim:
        kwargs['holo_dim'] = args.dim
    if getattr(args, 'embed_model', None):
        kwargs['embed_model'] = args.embed_model
    return HoloConfig(**kwargs)


def cmd_search(args):
    cfg = _load_config(args)
    rag = HoloRAG(cfg)
    query = ' '.join(args.query)

    if args.project:
        files = rag.search_scoped(query, args.project, top_k=args.top_k)
        for f in files:
            print(f"  {f.relevance*100:.1f}%  {f.filepath}")
    else:
        result = rag.search(query, top_k_projects=args.top_k)
        if args.json:
            out = {
                'query': result.query,
                'stage': result.stage,
                'projects': [
                    {
                        'name': p.name,
                        'similarity': round(p.similarity, 4),
                        'file_count': p.file_count,
                        'hits': p.hits,
                        'top_files': [
                            {'filepath': f.filepath, 'relevance': round(f.relevance, 4)}
                            for f in p.top_files
                        ],
                    }
                    for p in result.projects
                ],
            }
            print(json.dumps(out, indent=2))
        else:
            print(result.to_text())


def cmd_build(args):
    cfg = _load_config(args)
    rag = HoloRAG(cfg)
    print(f"Building holographic cache from {cfg.chroma_path}...")
    rag.build_cache(force=args.force)
    cache = rag.cache
    if cache.loaded:
        d = cache.data
        print(f"Done: {len(d['vectors'])} projects, {d['doc_count']} docs")
    else:
        print("Build failed — check logs.")
        sys.exit(1)


def cmd_projects(args):
    cfg = _load_config(args)
    rag = HoloRAG(cfg)
    projects = rag.list_projects()
    print(f"{len(projects)} projects:\n")
    for p in projects:
        print(f"  {p}")


def cmd_ingest(args):
    from fttf_holo_rag.ingest import ingest_directories

    cfg = _load_config(args)
    result = ingest_directories(
        args.directories,
        ollama_url=cfg.ollama_url,
        embed_model=cfg.embed_model,
        chroma_path=cfg.chroma_path,
        collection_name=cfg.chroma_collection,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        include_pdf=args.include_pdf,
        holo_cache_path=cfg.holo_cache_path,
    )
    print(
        f"Ingestion complete — added: {result.added}, "
        f"skipped (duplicate): {result.skipped}, "
        f"failed: {result.failed}"
    )
    if result.failed:
        sys.exit(1)


def cmd_prep(args):
    from fttf_holo_rag.chunker import prep_directory

    manifest = prep_directory(
        args.input_dir,
        args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        include_pdf=args.include_pdf,
    )
    print(
        f"prep complete — {len(manifest.chunks)} chunk(s) written to {manifest.output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog='fttf-holo',
        description='FTTF Holographic RAG — Fourier Transform holographic memory for local RAG',
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')

    # Global options
    parser.add_argument('--chroma-path', default=None, help='ChromaDB storage path')
    parser.add_argument('--collection', default=None, help='ChromaDB collection name')
    parser.add_argument('--ollama-url', default=None, help='Ollama embedding API URL')
    parser.add_argument('--cache-path', default=None, help='Holographic cache file path')
    parser.add_argument('--dim', type=int, default=None, help='Holographic dimension (default: 10000)')

    sub = parser.add_subparsers(dest='command', required=True)

    # search
    p_search = sub.add_parser('search', help='Search across projects')
    p_search.add_argument('query', nargs='+', help='Search query')
    p_search.add_argument('--project', '-p', default=None, help='Scope to a specific project')
    p_search.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    p_search.add_argument('--json', action='store_true', help='Output as JSON')
    p_search.set_defaults(func=cmd_search)

    # build
    p_build = sub.add_parser('build', help='Build/rebuild the holographic cache')
    p_build.add_argument('--force', '-f', action='store_true', help='Force full rebuild')
    p_build.set_defaults(func=cmd_build)

    # projects
    p_projects = sub.add_parser('projects', help='List all projects')
    p_projects.set_defaults(func=cmd_projects)

    # ingest
    p_ingest = sub.add_parser(
        'ingest',
        help='Crawl directories and ingest documents into ChromaDB',
    )
    p_ingest.add_argument('directories', nargs='+', help='Directories to crawl')
    p_ingest.add_argument('--batch-size', type=int, default=50, help='ChromaDB upsert batch size (default: 50)')
    p_ingest.add_argument('--max-chars', type=int, default=2000, help='Max characters per document (default: 2000)')
    p_ingest.add_argument('--include-pdf', action='store_true', help='Extract text from PDF files (requires PyMuPDF)')
    p_ingest.add_argument('--embed-model', default=None, help='Ollama embedding model name')
    p_ingest.set_defaults(func=cmd_ingest)

    # prep
    p_prep = sub.add_parser(
        'prep',
        help='Chunk large documents into smaller files ready for ingestion',
    )
    p_prep.add_argument('input_dir', help='Directory containing source documents')
    p_prep.add_argument('output_dir', help='Directory to write chunk files into')
    p_prep.add_argument('--chunk-size', type=int, default=25000, help='Max characters per chunk (default: 25000)')
    p_prep.add_argument('--overlap', type=int, default=500, help='Overlap between chunks in characters (default: 500)')
    p_prep.add_argument('--include-pdf', action='store_true', help='Extract text from PDF files (requires PyMuPDF)')
    p_prep.set_defaults(func=cmd_prep)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(name)s %(message)s')

    args.func(args)


if __name__ == '__main__':
    main()
