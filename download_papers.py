"""Download KV-cache sparsity papers from arXiv."""
import urllib.request
import os

papers = [
    ("1_H2O", "2306.14048", "H2O: Heavy-Hitter Oracle"),
    ("2_SnapKV", "2404.14469", "SnapKV: LLM Knows What You Are Looking for"),
    ("3_RocketKV", "2502.14051", "RocketKV: Accelerating Long-Context LLM Inference"),
    ("4_AttentionPredictor", "2502.04077", "AttentionPredictor: Temporal Patterns Matter"),
    ("5_SmallKV", "2508.02751", "SmallKV: Small Model Assisted Compensation"),
    ("7_MUSTAFAR", "2505.22913", "MUSTAFAR: Unstructured Sparsity for KV Cache"),
]

base = os.path.dirname(__file__)
for folder, arxiv_id, name in papers:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    path = os.path.join(base, "papers", folder, f"{arxiv_id}.pdf")
    print(f"Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path}")
    except Exception as e:
        print(f"  FAILED: {e}")

# CurDKV — search for its arXiv ID
print("\nCurDKV: need arXiv ID, checking...")
# Note: CurDKV doesn't have a public arXiv ID yet.
# Download from OpenReview if available, otherwise note it.
print("  CurDKV arXiv ID not found. Will check OpenReview.")

# Mnemosyne — local paper
print("\nMnemosyne: local paper, skipping download.")
