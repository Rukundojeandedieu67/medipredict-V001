"""Seed-held-out and time-purged training utilities for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class SeedPartitions:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def partition_seed_directories(seed_names: Iterable[str]) -> SeedPartitions:
    names = tuple(sorted(seed_names))
    if len(names) < 5:
        raise ValueError("At least five seed runs are required for train/validation/test splits")
    train_end = int(len(names) * 0.6)
    validation_end = int(len(names) * 0.8)
    if train_end < 1 or validation_end <= train_end or validation_end >= len(names):
        raise ValueError("Seed count cannot be divided into non-empty 60/20/20 partitions")
    return SeedPartitions(
        train=names[:train_end],
        validation=names[train_end:validation_end],
        test=names[validation_end:],
    )


def time_partition_masks(
    period_indices: np.ndarray,
    period_count: int,
    periods_per_day: int = 1,
    gap_days: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if period_count < 1 or periods_per_day < 1 or gap_days < 0:
        raise ValueError("period_count/periods_per_day must be positive and gap_days non-negative")
    train_end = int(period_count * 0.6)
    validation_end = int(period_count * 0.8)
    gap = gap_days * periods_per_day
    train = period_indices < train_end
    validation = (period_indices >= train_end + gap) & (period_indices < validation_end)
    test = (period_indices >= validation_end + gap) & (period_indices < period_count)
    return train, validation, test


def recall_at_daily_alert_budget(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    alerts_per_cluster: int,
) -> float:
    if alerts_per_cluster < 1:
        raise ValueError("alerts_per_cluster must be positive")
    true_positives = 0
    positive_total = int(np.sum(labels))
    if positive_total == 0:
        return 0.0
    for cluster in np.unique(cluster_ids):
        indices = np.flatnonzero(cluster_ids == cluster)
        selected_count = min(alerts_per_cluster, len(indices))
        selected = indices[np.argsort(-scores[indices], kind="stable")[:selected_count]]
        true_positives += int(np.sum(labels[selected]))
    return true_positives / positive_total


def selected_alert_indices(
    scores: np.ndarray, cluster_ids: np.ndarray, alerts_per_cluster: int
) -> np.ndarray:
    if alerts_per_cluster < 1:
        raise ValueError("alerts_per_cluster must be positive")
    selected = []
    for cluster in np.unique(cluster_ids):
        indices = np.flatnonzero(cluster_ids == cluster)
        count = min(alerts_per_cluster, len(indices))
        selected.extend(indices[np.argsort(-scores[indices], kind="stable")[:count]])
    return np.asarray(selected, dtype=np.int64)


def paired_bootstrap_recall_difference(
    labels: np.ndarray,
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    cluster_ids: np.ndarray,
    budgets: tuple[int, ...] = (1, 3, 5, 10),
    replicates: int = 200,
    seed: int = 2026,
) -> dict[str, list[float]]:
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    clusters = np.unique(cluster_ids)
    if len(clusters) < 2:
        raise ValueError("At least two seed-day clusters are required for paired bootstrap")
    cluster_members = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    total_positives = np.array(
        [labels[cluster_members[cluster]].sum() for cluster in clusters], dtype=float
    )
    model_true_positives: dict[int, np.ndarray] = {}
    baseline_true_positives: dict[int, np.ndarray] = {}
    for budget in budgets:
        model_counts = []
        baseline_counts = []
        for cluster in clusters:
            indices = cluster_members[cluster]
            model_selected = indices[
                np.argsort(-model_scores[indices], kind="stable")[: min(budget, len(indices))]
            ]
            baseline_selected = indices[
                np.argsort(-baseline_scores[indices], kind="stable")[: min(budget, len(indices))]
            ]
            model_counts.append(labels[model_selected].sum())
            baseline_counts.append(labels[baseline_selected].sum())
        model_true_positives[budget] = np.asarray(model_counts, dtype=float)
        baseline_true_positives[budget] = np.asarray(baseline_counts, dtype=float)

    rng = np.random.default_rng(seed)
    intervals = {}
    for budget in budgets:
        differences = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sample = rng.integers(0, len(clusters), size=len(clusters))
            positives = total_positives[sample].sum()
            if positives == 0:
                differences[replicate] = 0.0
                continue
            model_recall = model_true_positives[budget][sample].sum() / positives
            baseline_recall = baseline_true_positives[budget][sample].sum() / positives
            differences[replicate] = model_recall - baseline_recall
        intervals[str(budget)] = np.percentile(differences, [2.5, 97.5]).tolist()
    return intervals


def bootstrap_recall_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    budgets: tuple[int, ...] = (1, 3, 5, 10),
    replicates: int = 200,
    seed: int = 2026,
) -> dict[str, list[float]]:
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    clusters = np.unique(cluster_ids)
    if len(clusters) < 2:
        raise ValueError("At least two seed-day clusters are required for bootstrap")
    members = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    positives = np.asarray([labels[members[cluster]].sum() for cluster in clusters], dtype=float)
    true_positives: dict[int, np.ndarray] = {}
    for budget in budgets:
        true_positives[budget] = np.asarray(
            [
                labels[
                    members[cluster][
                        np.argsort(-scores[members[cluster]], kind="stable")[
                            : min(budget, len(members[cluster]))
                        ]
                    ]
                ].sum()
                for cluster in clusters
            ],
            dtype=float,
        )
    rng = np.random.default_rng(seed)
    intervals = {}
    for budget in budgets:
        sampled_recall = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sample = rng.integers(0, len(clusters), size=len(clusters))
            total_positives = positives[sample].sum()
            sampled_recall[replicate] = (
                true_positives[budget][sample].sum() / total_positives
                if total_positives
                else 0.0
            )
        intervals[str(budget)] = np.percentile(sampled_recall, [2.5, 97.5]).tolist()
    return intervals


def bootstrap_metric_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    alerts_per_cluster: int,
    replicates: int = 200,
    seed: int = 2026,
) -> dict[str, list[float]]:
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) < 2:
        raise ValueError("At least two independent seed-day clusters are required for bootstrap")
    rng = np.random.default_rng(seed)
    pr_auc_values = np.empty(replicates, dtype=float)
    recall_values = np.empty(replicates, dtype=float)
    cluster_indices = {
        cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique_clusters
    }
    for replicate in range(replicates):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled_parts = [cluster_indices[cluster] for cluster in sampled_clusters]
        indices = np.concatenate(sampled_parts)
        boot_clusters = np.concatenate(
            [np.full(len(part), position, dtype=np.int32) for position, part in enumerate(sampled_parts)]
        )
        boot_labels = labels[indices]
        boot_scores = scores[indices]
        pr_auc_values[replicate] = average_precision_score(boot_labels, boot_scores)
        recall_values[replicate] = recall_at_daily_alert_budget(
            boot_labels, boot_scores, boot_clusters, alerts_per_cluster
        )
    return {
        "pr_auc": np.percentile(pr_auc_values, [2.5, 97.5]).tolist(),
        "recall_at_alert_budget": np.percentile(recall_values, [2.5, 97.5]).tolist(),
    }
