"""Generate synthetic labels, compare models, and save the selected demo pipeline."""
from pathlib import Path

from med_reserve.modeling import train_and_evaluate


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    results = train_and_evaluate(project_dir)
    print("Synthetic demonstration model training complete.")
    print(f"Selected by 5-fold training CV macro-F1: {results['selected_model']}")
    print(f"Held-out synthetic test accuracy: {results['test_accuracy']:.3f}")
    print(f"Held-out synthetic test macro-F1: {results['test_macro_f1']:.3f}")
    print("Saved med_reserve_model.pkl and model_metrics.json.")
    print("These scores only measure reproduction of synthetic rule-generated labels.")

