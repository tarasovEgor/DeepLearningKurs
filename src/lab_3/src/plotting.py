import matplotlib.pyplot as plt

from pathlib import Path
from .utils import ensure_dir


def plot_main_comparison(summary_df, output_dir="outputs/figures"):
    ensure_dir(output_dir)

    quality_cols = ["correctness", "groundedness", "completeness", "coverage", "source_consistency", "rubric"]
    process_cols = ["n_steps", "latency"]

    plt.figure(figsize=(10, 5))
    summary_df[quality_cols].plot(kind="bar", figsize=(10, 5))
    plt.title("Сравнение качества результатов")
    plt.ylabel("Средний балл")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "quality_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    summary_df[process_cols].plot(kind="bar", figsize=(8, 4))
    plt.title("Сравнение процесса выполнения")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "process_comparison.png")
    plt.close()


def plot_factor_experiment(df, factor_col, metric_col, output_path):
    ensure_dir(str(Path(output_path).parent))
    agg = df.groupby(factor_col)[metric_col].mean()

    plt.figure(figsize=(8, 4))
    agg.plot(kind="bar")
    plt.title(f"{metric_col} vs {factor_col}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()