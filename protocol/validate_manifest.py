"""Offline validation for the frozen Phase 1 protocol."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

REQUIRED = ("protocol_version", "upstream", "model", "evaluation", "training", "resources", "selection", "analysis", "allowed_technical_fallbacks", "held_out_policy")
ALLOWED = {"single_oom_sequence_length_2048_to_1536_then_full_restart", "disable_optional_fused_kernels", "pin_compatible_package_version"}

def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []
    for key in REQUIRED:
        if key not in data or data[key] in (None, "", {}): errors.append(f"missing:{key}")
    if data.get("upstream", {}).get("commit") != "1899ad64fbfbc65790d259471cc4bf4de9437aa9": errors.append("upstream commit is not pinned")
    if data.get("model", {}).get("revision") != data.get("model", {}).get("tokenizer_revision"): errors.append("model/tokenizer revisions differ")
    decoding = data.get("evaluation", {}).get("decoding", {})
    for key in ("strategy", "temperature", "top_p", "seed", "batch_size", "auto_ceiling", "no_repeat_ngram"): 
        if key not in decoding or decoding[key] is None: errors.append(f"implicit decoding:{key}")
    fallbacks = data.get("allowed_technical_fallbacks", [])
    if set(fallbacks) - ALLOWED: errors.append("unauthorized fallback")
    if data.get("held_out_policy", {}).get("runner_preselection") != "deny_plaintext_outputs_metrics_and_aggregates": errors.append("held-out preselection access is not denied")
    if errors: raise ValueError("; ".join(errors))
    return data

def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default=Path(__file__).with_name("manifest.json"))
    args = parser.parse_args()
    load(args.manifest)
    print(f"valid manifest: {args.manifest} sha256={sha256(args.manifest)}")
