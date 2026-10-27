"""Dump cached file lists for target projects (for gold-set construction)."""
from fttf_holo_rag.cache import load_signed_cache

CACHE = r"C:\Users\akbon\Projects\AI_Command_Center\chroma_db_storage\holo_cache.pkl"
files = load_signed_cache(CACHE)["files"]
targets = ["lpf_pipeline", "lpf_data", "harmonic-field-consciousness",
           "consciousness-emergence-testbed", "qrng-analysis-toolkit",
           "ollama-auxrng", "helios-trajectory-analysis", "lgi-executive",
           "FTTF-fournier (1)", "FTTF-fournier (2)", "BitNet", "quantum-llama.cpp",
           "mcmc-consciousness-lab", "qrng-diagnostic-lab", "LlamaFactory",
           "crypto-ml-trading-system"]
for t in targets:
    print("=== " + t + " ===")
    fl = files.get(t, [])
    for f in fl[:40]:
        print("   ", f)
    if len(fl) > 40:
        print(f"    ... ({len(fl)} total)")
