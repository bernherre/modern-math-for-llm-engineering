"""Run all optional real-sentence Hugging Face experiments."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "geometry/metric_rag/example_hf.py",
    "geometry/information_geometry_finetuning/example_hf.py",
    "topology/sheaf_rag/example_hf.py",
    "algebra/lie_algebra_finetuning/example_hf.py",
    "algebra/noncommutative_adapter_composition/example_hf.py",
    "transport/optimal_transport_reranking/example_hf.py",
    "probability/high_dimensional_quantization_rag/example_hf.py",
]

for rel in EXAMPLES:
    print("\n" + "=" * 80)
    print(rel)
    print("=" * 80)
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("\nAll Hugging Face sentence-embedding labs completed successfully.")
