from pathlib import Path

from src.capstone_analysis import run_full_analysis


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    result = run_full_analysis(project_root)
    print("Rows analyzed:", result["rows"])
    print("High class rate:", f"{result['high_class_rate']:.3f}")
    print("Best model:", result["best_model"])
    print("Best k:", result["best_k"])
