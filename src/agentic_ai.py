# src/agentic_ai.py
from typing import List
import os

# Try to use HF summarizer (small model) if installed, else fallback
try:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1)
    _hf_ok = True
except Exception:
    summarizer = None
    _hf_ok = False

def summarize_logs(log_texts: List[str], max_input_chars: int = 1500) -> str:
    """
    Concatenate logs and produce a short summary.
    If HF summarizer available, use it; else produce a simple heuristic summary.
    """
    if not log_texts:
        return "No logs to summarize."
    combined = "\n".join(log_texts)
    combined = combined[:max_input_chars]
    if _hf_ok and summarizer is not None:
        try:
            out = summarizer(combined, max_length=120, min_length=30, do_sample=False)
            return out[0]["summary_text"]
        except Exception as e:
            # fallback
            pass
    # heuristic: count labels/protocol mentions and return short bullets
    lines = combined.splitlines()
    n = len(lines)
    proto_counts = {}
    for l in lines:
        if "tcp" in l.lower():
            proto_counts["tcp"] = proto_counts.get("tcp", 0) + 1
        if "udp" in l.lower():
            proto_counts["udp"] = proto_counts.get("udp", 0) + 1
    bullets = []
    bullets.append(f"Total logs considered: {n}.")
    if proto_counts:
        bullets.append("Protocol counts: " + ", ".join(f"{k}:{v}" for k,v in proto_counts.items()))
    # highlight frequent IPs if present
    ips = []
    import re
    for l in lines:
        found = re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", l)
        ips.extend(found)
    if ips:
        from collections import Counter
        c = Counter(ips).most_common(3)
        bullets.append("Top IPs: " + ", ".join(f"{ip}({cnt})" for ip, cnt in c))
    return " ".join(bullets)
