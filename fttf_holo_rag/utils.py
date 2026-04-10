"""
utils.py — Shared helpers for the holographic RAG system.

No external framework dependencies.
"""


def extract_project_name(filepath: str, allowed_roots: list[str] | None = None) -> str:
    """Extract the project name from a filepath.

    Uses allowed_roots when the path matches a known root,
    otherwise falls back to path-pattern heuristics.
    """
    fp = filepath.replace('\\', '/')
    roots = [r.replace('\\', '/').rstrip('/') for r in (allowed_roots or [])]
    for root in roots:
        if fp.startswith(root + '/') or fp == root:
            rest = fp[len(root):].lstrip('/')
            parts = rest.split('/')
            return parts[0] if parts and parts[0] else 'unknown'

    # Fallback: parse well-known path shapes
    parts = [p for p in fp.split('/') if p]
    if not parts:
        return 'root'
    # Strip drive letter (E:, C:, H:)
    if parts[0].upper().endswith(':'):
        parts = parts[1:]
    if not parts:
        return 'root'
    # C:\Users\<user>\Projects\<project>
    if len(parts) >= 4 and parts[0].lower() == 'users' and parts[2].lower() == 'projects':
        return parts[3]
    # C:\Users\<user>\OneDrive\Documents\GitHub\<project>
    if (len(parts) >= 6 and parts[0].lower() == 'users'
            and parts[3].lower() == 'documents' and parts[4].lower() == 'github'):
        return parts[5]
    # E:\<project>\... or H:\<project>\...
    return parts[0]
