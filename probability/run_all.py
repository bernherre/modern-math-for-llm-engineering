from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "geometry/metric_rag/example.py",
    "geometry/information_geometry_finetuning/example.py",
    "topology/sheaf_rag/example.py",
    "algebra/lie_algebra_finetuning/example.py",
    "algebra/noncommutative_adapter_composition/example.py",
    "transport/optimal_transport_reranking/example.py",
    "probability/high_dimensional_quantization_rag/example.py",
    "retrieval/neighborhood_rerank_optimizer/example.py",
    "combinatorics/matroid_selection/example.py",
]

for rel in EXAMPLES:
    print("\n" + "=" * 80)
    print(rel)
    print("=" * 80)
    subprocess.run([sys.executable, str(ROOT / rel)], check=True)

print("\nAll lightweight NumPy labs completed successfully.")
