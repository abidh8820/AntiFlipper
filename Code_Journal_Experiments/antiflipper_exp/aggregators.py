from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import ANTIFLIPPER_METHODS


State = dict[str, torch.Tensor]


def _float_keys(state: State) -> list[str]:
    return [key for key, value in state.items() if torch.is_floating_point(value)]


def weighted_average(states: list[State], weights: np.ndarray | list[float]) -> State:
    if not states:
        raise ValueError("Cannot aggregate zero states")
    w = torch.as_tensor(weights, dtype=torch.float64)
    if len(w) != len(states) or not torch.isfinite(w).all() or float(w.sum()) <= 0:
        w = torch.ones(len(states), dtype=torch.float64)
    w /= w.sum()
    result = copy.deepcopy(states[0])
    for key in result:
        if not torch.is_floating_point(result[key]):
            continue
        acc = torch.zeros_like(result[key])
        for index, state in enumerate(states):
            acc.add_(state[key], alpha=float(w[index]))
        result[key] = acc
    return result


def coordinate_median(states: list[State]) -> State:
    result = copy.deepcopy(states[0])
    for key in _float_keys(result):
        result[key] = torch.stack([state[key] for state in states]).median(dim=0).values
    return result


def trimmed_mean(states: list[State], trim_fraction: float) -> State:
    n = len(states)
    trim = int(n * trim_fraction)
    if 2 * trim >= n:
        raise ValueError(f"Trimmed mean needs 2*floor(n*trim_fraction) < n; n={n}, trim={trim}")
    result = copy.deepcopy(states[0])
    for key in _float_keys(result):
        stacked = torch.stack([state[key] for state in states])
        ordered = torch.sort(stacked, dim=0).values
        result[key] = ordered[trim:n - trim].mean(dim=0) if trim else ordered.mean(dim=0)
    return result


def flatten_delta(state: State, global_state: State, keys: tuple[str, ...] | None = None) -> torch.Tensor:
    if keys is None:
        keys = tuple(_float_keys(global_state))
    chunks = [(state[k].detach().float().cpu() - global_state[k].detach().float().cpu()).reshape(-1) for k in keys]
    return torch.cat(chunks)


def multi_krum(states: list[State], global_state: State, f: int) -> tuple[State, np.ndarray]:
    n = len(states)
    if n <= 2 * f + 2:
        raise ValueError(f"Multi-Krum assumptions violated: n={n}, f={f}")
    vectors = torch.stack([flatten_delta(x, global_state) for x in states])
    distances = torch.cdist(vectors, vectors).pow(2)
    neighbor_count = n - f - 2
    scores = torch.topk(distances, k=neighbor_count + 1, largest=False).values[:, 1:].sum(dim=1)
    selected_count = max(1, n - f - 2)
    selected = torch.argsort(scores)[:selected_count].cpu().numpy()
    weights = np.zeros(n)
    weights[selected] = 1
    return weighted_average(states, weights), weights


def _foolsgold_weights(memory: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(memory)
    if n == 1:
        return np.ones(1)
    cs = cosine_similarity(memory) - np.eye(n)
    max_cs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i != j and max_cs[i] < max_cs[j] and max_cs[j] > 0:
                cs[i, j] *= max_cs[i] / max_cs[j]
    weights = np.clip(1 - np.max(cs, axis=1), 0, 1)
    maximum = weights.max(initial=0)
    if maximum <= 0:
        return np.ones(n)
    weights /= maximum
    weights[weights == 1] = 0.99
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.log(weights / (1 - weights)) + 0.5
    return np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)


@dataclass
class AggregatorState:
    foolsgold_memory: dict[int, np.ndarray] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {"foolsgold_memory": self.foolsgold_memory}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        self.foolsgold_memory = value.get("foolsgold_memory", {})


def foolsgold(states: list[State], global_state: State, client_ids: list[int], memory_state: AggregatorState) -> tuple[State, np.ndarray]:
    deltas = [flatten_delta(state, global_state).numpy() for state in states]
    for client_id, delta in zip(client_ids, deltas):
        memory_state.foolsgold_memory[client_id] = memory_state.foolsgold_memory.get(client_id, np.zeros_like(delta)) + delta
    matrix = np.stack([memory_state.foolsgold_memory[client_id] for client_id in client_ids])
    weights = _foolsgold_weights(matrix)
    return weighted_average(states, weights), weights


def tolpegin(states: list[State], global_state: State) -> tuple[State, np.ndarray]:
    """PCA/KMeans last-layer detector ported from the submitted repository."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    candidates = [key for key in _float_keys(global_state) if state_ndim(global_state[key]) >= 2]
    key = candidates[-1]
    deltas = np.stack([(global_state[key] - state[key]).detach().cpu().numpy() for state in states])
    class_distances: list[float] = []
    class_labels: list[np.ndarray] = []
    for cls in range(deltas.shape[1] if deltas.ndim > 2 else 1):
        features = deltas[:, cls].reshape(len(states), -1) if deltas.ndim > 2 else deltas.reshape(len(states), -1)
        features = StandardScaler().fit_transform(features)
        components = max(1, min(features.shape[0] - 1, features.shape[1], 8))
        reduced = PCA(n_components=components).fit_transform(features)
        fitted = KMeans(n_clusters=2, random_state=0, n_init=10).fit(reduced)
        centers = fitted.cluster_centers_
        class_distances.append(float(np.square(centers[0] - centers[1]).sum()))
        class_labels.append(fitted.labels_)
    labels = class_labels[int(np.argmax(class_distances))]
    benign_label = 0 if np.sum(labels == 0) >= np.sum(labels == 1) else 1
    weights = (labels == benign_label).astype(float)
    return weighted_average(states, weights), weights


def state_ndim(value: torch.Tensor) -> int:
    return len(value.shape)


def lfighter(states: list[State], global_state: State) -> tuple[State, np.ndarray]:
    """Last-layer two-cluster LFighter-style aggregation used by the original code."""
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    matrix_keys = [k for k in _float_keys(global_state) if state_ndim(global_state[k]) == 2]
    bias_keys = [k for k in _float_keys(global_state) if state_ndim(global_state[k]) == 1]
    w_key, b_key = matrix_keys[-1], bias_keys[-1]
    dw = np.stack([(global_state[w_key] - s[w_key]).cpu().numpy() for s in states])
    db = np.stack([(global_state[b_key] - s[b_key]).cpu().numpy() for s in states])
    frequency = np.linalg.norm(dw, axis=-1).sum(axis=0) + np.abs(db).sum(axis=0)
    classes = frequency.argsort()[-2:]
    features = dw[:, classes].reshape(len(states), -1)
    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(features)
    clusters = [features[labels == label] for label in (0, 1)]
    dissimilarity = []
    for cluster in clusters:
        if len(cluster) < 2:
            dissimilarity.append(float("inf"))
        else:
            similarity = cosine_similarity(cluster) - np.eye(len(cluster))
            dissimilarity.append((len(cluster) / len(states)) * (1 - np.mean(np.min(similarity, axis=1))))
    # Preserve the original implementation's selection rule.
    benign_label = 1 if dissimilarity[0] < dissimilarity[1] else 0
    weights = (labels == benign_label).astype(float)
    return weighted_average(states, weights), weights


def flame(states: list[State], global_state: State, seed: int, parameter_keys: tuple[str, ...]) -> tuple[State, np.ndarray]:
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=len(states) // 2 + 1, min_samples=1, allow_single_cluster=True)
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=len(states) // 2 + 1, min_samples=1, allow_single_cluster=True)
        except ImportError as exc:
            raise RuntimeError("FLAME requires either hdbscan or scikit-learn with sklearn.cluster.HDBSCAN") from exc
    from sklearn.metrics.pairwise import cosine_similarity

    deltas = np.stack([flatten_delta(state, global_state, parameter_keys).numpy() for state in states])
    similarities = cosine_similarity(deltas)
    labels = clusterer.fit_predict(similarities)
    benign = np.arange(len(states)) if np.all(labels == -1) else np.flatnonzero(labels != -1)
    norms = np.linalg.norm(deltas, axis=1)
    clip = float(np.median(norms))
    parameters = set(parameter_keys)
    buffer_keys = [key for key in _float_keys(global_state) if key not in parameters]
    clipped: list[State] = []
    for index in benign:
        scale = min(1.0, clip / max(float(norms[index]), 1e-12))
        value = dict(global_state)
        # Average accepted clients' floating buffers without clipping or adding noise.
        value.update({key: states[index][key] for key in buffer_keys})
        for key in parameter_keys:
            value[key] = global_state[key] + (states[index][key] - global_state[key]) * scale
        clipped.append(value)
    result = weighted_average(clipped, np.ones(len(clipped)))
    sigma = 0.001 * clip
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for key in parameter_keys:
        noise = torch.randn(result[key].shape, generator=generator, dtype=result[key].dtype) * sigma
        result[key] += noise.to(result[key].device)
    weights = np.zeros(len(states)); weights[benign] = 1
    return result, weights


def update_consistency(states: list[State], global_state: State, z_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """AntiFLipper-CC cross-check of the self-reported accuracy.

    A label flip must move the classifier rows of the swapped classes, so a client
    that trained on flipped labels has a last-layer update pointing away from the
    honest majority even if it lies about its accuracy. Each update is compared with
    the coordinate-wise median last-layer update (no server data needed). Clients
    whose cosine similarity is a robust (median/MAD) outlier on the low side are
    marked inconsistent. Cost is O(n * |last layer|), negligible next to clustering.

    Returns (similarities, inconsistent_mask).
    """
    matrix_keys = [k for k in _float_keys(global_state) if state_ndim(global_state[k]) == 2]
    bias_keys = [k for k in _float_keys(global_state) if state_ndim(global_state[k]) == 1]
    keys = [matrix_keys[-1]] + ([bias_keys[-1]] if bias_keys else [])
    deltas = torch.stack([
        torch.cat([(s[k].detach().float().cpu() - global_state[k].detach().float().cpu()).reshape(-1) for k in keys])
        for s in states
    ])
    reference = deltas.median(dim=0).values
    similarity = torch.nn.functional.cosine_similarity(deltas, reference.unsqueeze(0).expand_as(deltas), dim=1, eps=1e-12).numpy()
    if len(states) < 3:
        return similarity, np.zeros(len(states), dtype=bool)
    center = float(np.median(similarity))
    mad = float(np.median(np.abs(similarity - center)))
    scale = 1.4826 * mad + 1e-6
    robust_z = (similarity - center) / scale
    return similarity, robust_z < -z_cut


def fltrust(states: list[State], global_state: State, root_state: State,
            parameter_keys: tuple[str, ...] | None = None) -> tuple[State, np.ndarray]:
    """FLTrust. Trust scores and norm rescaling use trainable parameters only.

    Rescaling every floating tensor (the earlier version) also rescaled BatchNorm
    running-variance updates by root_norm / client_norm, which can drive a variance
    negative and produced NaN test losses on ResNet-18. Floating buffers are now the
    score-weighted average of the clients' buffers, which keeps variances positive.
    """
    keys = tuple(parameter_keys) if parameter_keys else tuple(_float_keys(global_state))
    root_delta = flatten_delta(root_state, global_state, keys)
    root_norm = torch.linalg.vector_norm(root_delta).clamp_min(1e-12)
    client_deltas = [flatten_delta(state, global_state, keys) for state in states]
    scores = np.asarray([max(0.0, float(torch.dot(delta, root_delta) / (torch.linalg.vector_norm(delta).clamp_min(1e-12) * root_norm))) for delta in client_deltas])
    if scores.sum() <= 0:
        return copy.deepcopy(global_state), scores
    result = copy.deepcopy(global_state)
    scores /= scores.sum()
    client_norms = [torch.linalg.vector_norm(delta).clamp_min(1e-12) for delta in client_deltas]
    for key in keys:
        aggregate = torch.zeros_like(result[key])
        for score, state, delta_norm in zip(scores, states, client_norms):
            delta = state[key] - global_state[key]
            aggregate.add_(delta * (root_norm / delta_norm), alpha=float(score))
        result[key] = global_state[key] + aggregate
    parameters = set(keys)
    for key in _float_keys(result):
        if key not in parameters:
            result[key] = sum(float(score) * state[key] for score, state in zip(scores, states))
    return result, scores


def aggregate(
    method: str,
    states: list[State],
    global_state: State,
    client_ids: list[int],
    malicious_fraction: float,
    runtime_state: AggregatorState,
    *,
    trust_weights: list[float] | None = None,
    root_state: State | None = None,
    seed: int = 0,
    parameter_keys: tuple[str, ...] | None = None,
) -> tuple[State, np.ndarray]:
    n = len(states)
    if method == "fedavg":
        weights = np.ones(n); return weighted_average(states, weights), weights
    if method == "fedavg_weighted":
        # Original FedAvg (McMahan et al.): weight each client by its local sample count.
        weights = np.asarray(trust_weights, dtype=float); return weighted_average(states, weights), weights
    if method in ANTIFLIPPER_METHODS:
        weights = np.asarray(trust_weights, dtype=float); return weighted_average(states, weights), weights
    if method == "median":
        return coordinate_median(states), np.ones(n)
    if method == "trimmed_mean":
        return trimmed_mean(states, malicious_fraction), np.ones(n)
    if method == "multi_krum":
        return multi_krum(states, global_state, int(malicious_fraction * n))
    if method == "foolsgold":
        return foolsgold(states, global_state, client_ids, runtime_state)
    if method == "tolpegin":
        return tolpegin(states, global_state)
    if method == "lfighter":
        return lfighter(states, global_state)
    if method == "flame":
        if not parameter_keys:
            raise ValueError("FLAME requires trainable parameter names")
        return flame(states, global_state, seed, parameter_keys)
    if method == "fltrust":
        if root_state is None:
            raise ValueError("FLTrust requires a root update")
        return fltrust(states, global_state, root_state, parameter_keys)
    raise ValueError(f"Unknown aggregation method: {method}")
