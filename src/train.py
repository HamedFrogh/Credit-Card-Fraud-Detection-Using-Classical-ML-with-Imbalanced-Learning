import argparse
from pathlib import Path

from src.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fraud detection models on imbalanced data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the Kaggle credit card fraud CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where results and plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_experiment(data_path=args.data_path, output_dir=output_dir)

    print("Baseline model comparison (sorted by F1):")
    print(results_df.to_string(index=False))
    print(f"\nSaved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
