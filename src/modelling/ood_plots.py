from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_ood_curves(
    y_true: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str,
    fold_id: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, ood_scores)
    roc_auc = float(auc(fpr, tpr))

    precision, recall, _ = precision_recall_curve(y_true, ood_scores)
    pr_auc = float(auc(recall, precision))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_title(f"{method_name} | fold {fold_id} | ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"AUC={pr_auc:.4f}")
    axes[1].set_title(f"{method_name} | fold {fold_id} | PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"{method_name}_fold{fold_id:02d}_roc_pr.png", dpi=220)
    plt.close(fig)


def plot_score_distribution(
    y_true: np.ndarray,
    ood_scores: np.ndarray,
    method_name: str,
    fold_id: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "score": ood_scores,
            "group": np.where(y_true == 1, "OOD", "ID"),
        }
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(
        data=df,
        x="score",
        hue="group",
        fill=True,
        common_norm=False,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title(f"{method_name} | fold {fold_id} | score distribution")
    ax.set_xlabel("OOD score (higher => more OOD)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{method_name}_fold{fold_id:02d}_score_dist.png", dpi=220)
    plt.close(fig)


def plot_fold_metrics_summary(
    results_df: pd.DataFrame,
    out_dir: Path,
    metrics: tuple[str, ...] = ("auroc", "f1", "accuracy"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for metric in metrics:
        if metric not in results_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=results_df, x="method", y=metric, ax=ax)
        sns.stripplot(
            data=results_df,
            x="method",
            y=metric,
            color="black",
            alpha=0.55,
            ax=ax,
        )
        ax.set_title(f"OOD {metric.upper()} across folds")
        fig.tight_layout()
        fig.savefig(out_dir / f"summary_{metric}.png", dpi=220)
        plt.close(fig)
