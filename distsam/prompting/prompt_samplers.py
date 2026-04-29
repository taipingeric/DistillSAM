import re
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np


# Strategy notes for DistillSAM teacher prompt sampling.
# fg_random_pos_N: positive point prompts from the foreground union; no
# negative points and no box prompt. budget_units=N, sparse-KD compatible
# when N equals num_prompt_tokens.
# fg_class_balanced_pos: positive point prompts split evenly across present
# foreground classes; no negative points or box. budget_units=num_prompt_tokens.
# fg_area_proportional_pos: positive point prompts split by foreground class
# pixel area, with present classes covered when possible; no negative points
# or box. budget_units=num_prompt_tokens.
# fg_connected_component_balanced_pos: positive point prompts spread across
# foreground connected components; no negative points or box.
# fg_distance_center_pos: positive point prompts sampled from distance-transform
# interior regions; no negative points or box.
# fg_mixed_center_boundary_pos: positive point prompts sampled half from
# interior regions and half from boundary-near foreground; no negative points
# or box.
# fg_pos75_bgneg25: positive foreground and negative background point prompts;
# no box. budget_units=num_prompt_tokens.
# fg_class_balanced_pos75_bgneg25: class-balanced positive points plus negative
# background points; no box. budget_units=num_prompt_tokens.
# fg_box / fg_box_points: foreground union box, optionally with positive
# points. These are oracle upper bounds; legacy budget counts are preserved.
# fg_box_*_pos8_neg6 / fg_expandbox_pos8_neg6: a SAM box prompt plus positive
# and negative points. The box counts as 2 sparse tokens; points consume the
# remaining budget. Negative points are sampled from non-mask regions inside
# the box first, then expanded/boundary/global fallbacks. sparse-KD compatible
# when budget_units equals num_prompt_tokens.
# fg_top2box_pos8_neg4: two component boxes when available; boxes count as
# 4 sparse tokens total. Remaining budget is split into positive and negative
# points. If only one component exists, it falls back to one-box behavior.
# pc_*_each strategies repeat the corresponding prompt design separately for
# each present foreground class. Per-class box+point strategies use class boxes,
# class positive points, and negatives from non-class regions; these are useful
# oracle/class-separation probes and are not the default sparse-KD teacher.
FOREGROUND_STRATEGIES = [
    "fg_random_pos_4",
    "fg_random_pos_8",
    "fg_random_pos_16",
    "fg_random_pos_32",
    "fg_class_balanced_pos",
    "fg_area_proportional_pos",
    "fg_connected_component_balanced_pos",
    "fg_distance_center_pos",
    "fg_mixed_center_boundary_pos",
    "fg_pos75_bgneg25",
    "fg_class_balanced_pos75_bgneg25",
    "fg_box",
    "fg_box_points",
    "fg_box_pos8_neg6",
    "fg_box_classbalanced_pos8_neg6",
    "fg_box_area_proportional_pos8_neg6",
    "fg_box_center_pos8_neg6",
    "fg_box_mixed_pos8_neg6",
    "fg_expandbox_pos8_neg6",
    "fg_top2box_pos8_neg4",
]

PER_CLASS_STRATEGIES = [
    "pc_random_pos_each",
    "pc_class_pos75_neg25_each",
    "pc_distance_center_pos_each",
    "pc_mixed_center_boundary_pos_each",
    "pc_box_each",
    "pc_box_points_each",
    "pc_box_pos8_neg6_each",
    "pc_box_center_pos8_neg6_each",
    "pc_box_otherfg_neg6_each",
    "pc_expandbox_pos8_neg6_each",
]

BOX_STRATEGIES = [
    "fg_box",
    "fg_box_points",
    "pc_box_each",
    "pc_box_points_each",
    "fg_box_pos8_neg6",
    "fg_box_classbalanced_pos8_neg6",
    "fg_box_area_proportional_pos8_neg6",
    "fg_box_center_pos8_neg6",
    "fg_box_mixed_pos8_neg6",
    "fg_expandbox_pos8_neg6",
    "fg_top2box_pos8_neg4",
    "pc_box_pos8_neg6_each",
    "pc_box_center_pos8_neg6_each",
    "pc_box_otherfg_neg6_each",
    "pc_expandbox_pos8_neg6_each",
]
BOX_POINT_STRATEGIES = [
    "fg_box_pos8_neg6",
    "fg_box_classbalanced_pos8_neg6",
    "fg_box_area_proportional_pos8_neg6",
    "fg_box_center_pos8_neg6",
    "fg_box_mixed_pos8_neg6",
    "fg_expandbox_pos8_neg6",
    "fg_top2box_pos8_neg4",
    "pc_box_pos8_neg6_each",
    "pc_box_center_pos8_neg6_each",
    "pc_box_otherfg_neg6_each",
    "pc_expandbox_pos8_neg6_each",
]
ALL_STRATEGIES = FOREGROUND_STRATEGIES + PER_CLASS_STRATEGIES


def _strategy_budget(strategy: str, total_budget: int) -> int:
    match = re.search(r"_(\d+)$", strategy)
    if match is not None:
        return int(match.group(1))
    return int(total_budget)


def _empty_prompt(strategy: str, kd_compatible: bool, budget_units: int, reason: str) -> Dict:
    return {
        "point_coords": None,
        "point_labels": None,
        "box": None,
        "boxes": None,
        "meta": {
            "strategy": strategy,
            "kd_compatible": kd_compatible,
            "sparse_kd_compatible": kd_compatible,
            "oracle_upper_bound": not kd_compatible,
            "prompt_type": "point",
            "budget_units": budget_units,
            "num_pos_points": 0,
            "num_neg_points": 0,
            "has_box": False,
            "skip_reason": reason,
        },
    }


def _valid_foreground_classes(
    semantic_mask: np.ndarray,
    foreground_classes: Iterable[int],
    ignore_index: int,
) -> List[int]:
    valid = semantic_mask != ignore_index
    return [
        int(class_id)
        for class_id in foreground_classes
        if np.any((semantic_mask == int(class_id)) & valid)
    ]


def sample_points_from_mask(
    mask: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    replace_if_needed: bool = True,
) -> Optional[np.ndarray]:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    replace = replace_if_needed and len(xs) < num_points
    if not replace and len(xs) < num_points:
        num_points = len(xs)
    indices = rng.choice(len(xs), size=num_points, replace=replace)
    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def _sample_from_mask(mask: np.ndarray, num_points: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    return sample_points_from_mask(mask, num_points, rng, replace_if_needed=True)


def _sample_top_distance(mask: np.ndarray, num_points: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if not np.any(mask):
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = dist[mask]
    if values.size == 0:
        return _sample_from_mask(mask, num_points, rng)
    threshold = np.percentile(values, 70.0)
    candidate = mask & (dist >= threshold)
    if not np.any(candidate):
        candidate = mask
    return _sample_from_mask(candidate, num_points, rng)


def _sample_boundary_inside(mask: np.ndarray, num_points: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if not np.any(mask):
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    values = dist[mask]
    if values.size == 0:
        return _sample_from_mask(mask, num_points, rng)
    threshold = max(1.5, float(np.percentile(values, 35.0)))
    candidate = mask & (dist <= threshold)
    if not np.any(candidate):
        candidate = mask
    return _sample_from_mask(candidate, num_points, rng)


def compute_bbox(binary_mask: np.ndarray, expand_ratio: float = 0.0, image_shape=None) -> Optional[np.ndarray]:
    mask = binary_mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())
    if expand_ratio > 0:
        width = max(1.0, x2 - x1 + 1.0)
        height = max(1.0, y2 - y1 + 1.0)
        dx = width * float(expand_ratio)
        dy = height * float(expand_ratio)
        x1 -= dx
        x2 += dx
        y1 -= dy
        y2 += dy
        if image_shape is not None:
            h, w = image_shape[:2]
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(w - 1), x2)
            y2 = min(float(h - 1), y2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    return compute_bbox(mask)


def _mask_from_bbox(bbox: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    mask = np.zeros((h, w), dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def allocate_pos_neg_budget(num_prompt_tokens: int, box_tokens: int = 2, pos_ratio: float = 8.0 / 14.0):
    remaining = max(0, int(num_prompt_tokens) - int(box_tokens))
    if remaining <= 1:
        return max(1, remaining), 0
    pos_count = int(round(remaining * float(pos_ratio)))
    pos_count = max(1, min(remaining - 1, pos_count))
    neg_count = remaining - pos_count
    neg_count = max(1, neg_count)
    if pos_count + neg_count > remaining:
        pos_count = max(1, remaining - neg_count)
    return pos_count, neg_count


def sample_negative_in_box(
    bbox,
    positive_mask: np.ndarray,
    semantic_mask: np.ndarray,
    ignore_index: int,
    num_points: int,
    rng: np.random.Generator,
    prefer_boundary: bool = True,
    fallback_global: bool = True,
) -> Optional[np.ndarray]:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    valid = semantic_mask != ignore_index
    box_mask = _mask_from_bbox(np.asarray(bbox), semantic_mask.shape)
    base_candidates = box_mask & (~positive_mask) & valid
    candidate_masks = []
    if prefer_boundary and np.any(base_candidates):
        kernel = np.ones((9, 9), dtype=np.uint8)
        near = cv2.dilate(positive_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        near_candidates = base_candidates & near
        if np.any(near_candidates):
            candidate_masks.append(near_candidates)
    candidate_masks.append(base_candidates)
    expanded_bbox = compute_bbox(box_mask, expand_ratio=0.10, image_shape=semantic_mask.shape)
    if expanded_bbox is not None:
        candidate_masks.append(_mask_from_bbox(expanded_bbox, semantic_mask.shape) & (~positive_mask) & valid)
    if fallback_global:
        kernel = np.ones((9, 9), dtype=np.uint8)
        near = cv2.dilate(positive_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        candidate_masks.append(near & (~positive_mask) & valid)
        candidate_masks.append((~positive_mask) & valid)

    sampled = []
    used = np.zeros_like(positive_mask, dtype=bool)
    remaining = int(num_points)
    last_nonempty = None
    for candidates in candidate_masks:
        candidates = candidates & (~used)
        if not np.any(candidates):
            continue
        last_nonempty = candidates
        count = int(candidates.sum())
        take = min(remaining, count)
        points = sample_points_from_mask(candidates, take, rng, replace_if_needed=False)
        if points is not None and len(points) > 0:
            sampled.append(points)
            used[points[:, 1].astype(np.int64), points[:, 0].astype(np.int64)] = True
            remaining -= len(points)
        if remaining <= 0:
            break
    if remaining > 0 and last_nonempty is not None:
        extra = sample_points_from_mask(last_nonempty, remaining, rng, replace_if_needed=True)
        if extra is not None:
            sampled.append(extra)
    return _concat_points(sampled)


def _concat_points(point_sets: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [points for points in point_sets if points is not None and len(points) > 0]
    if not valid:
        return None
    return np.concatenate(valid, axis=0).astype(np.float32)


def _allocate_balanced(classes: List[int], total: int) -> Dict[int, int]:
    if total <= 0 or not classes:
        return {class_id: 0 for class_id in classes}
    base = total // len(classes)
    remainder = total % len(classes)
    return {
        class_id: base + (1 if idx < remainder else 0)
        for idx, class_id in enumerate(classes)
    }


def _allocate_area(semantic_mask: np.ndarray, classes: List[int], total: int) -> Dict[int, int]:
    if total <= 0 or not classes:
        return {class_id: 0 for class_id in classes}
    areas = np.array([np.sum(semantic_mask == class_id) for class_id in classes], dtype=np.float64)
    if areas.sum() <= 0:
        return _allocate_balanced(classes, total)
    if total < len(classes):
        order = np.argsort(-areas)
        selected = set(int(idx) for idx in order[:total])
        return {
            class_id: (1 if idx in selected else 0)
            for idx, class_id in enumerate(classes)
        }
    alloc = np.floor(total * areas / areas.sum()).astype(np.int64)
    alloc = np.maximum(alloc, 1)
    while int(alloc.sum()) > total:
        idx = int(np.argmax(alloc))
        if alloc[idx] <= 1:
            break
        alloc[idx] -= 1
    while int(alloc.sum()) < total:
        frac = total * areas / areas.sum() - alloc
        alloc[int(np.argmax(frac))] += 1
    return {class_id: int(count) for class_id, count in zip(classes, alloc)}


def _sample_class_points(
    semantic_mask: np.ndarray,
    classes: List[int],
    allocation: Dict[int, int],
    rng: np.random.Generator,
    mode: str = "random",
) -> Optional[np.ndarray]:
    point_sets = []
    for class_id in classes:
        class_mask = semantic_mask == class_id
        count = int(allocation.get(class_id, 0))
        if mode == "center":
            points = _sample_top_distance(class_mask, count, rng)
        elif mode == "boundary":
            points = _sample_boundary_inside(class_mask, count, rng)
        else:
            points = _sample_from_mask(class_mask, count, rng)
        point_sets.append(points)
    return _concat_points(point_sets)


def _sample_negative_near_boundary(
    target_mask: np.ndarray,
    semantic_mask: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    ignore_index: int,
    prefer_other_foreground: bool = False,
    background_id: int = 0,
) -> Optional[np.ndarray]:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    valid = semantic_mask != ignore_index
    if prefer_other_foreground:
        candidates = valid & (~target_mask) & (semantic_mask != background_id)
        if np.any(candidates):
            return _sample_from_mask(candidates, num_points, rng)
    kernel = np.ones((9, 9), dtype=np.uint8)
    near = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    candidates = near & (~target_mask) & valid & (semantic_mask == background_id)
    if not np.any(candidates):
        candidates = (~target_mask) & valid & (semantic_mask == background_id)
    if not np.any(candidates):
        candidates = (~target_mask) & valid
    return _sample_from_mask(candidates, num_points, rng)


def _sample_connected_component_points(
    foreground_mask: np.ndarray,
    total_budget: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if total_budget <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    num_labels, labels = cv2.connectedComponents(foreground_mask.astype(np.uint8), connectivity=8)
    component_ids = [idx for idx in range(1, num_labels) if np.any(labels == idx)]
    if not component_ids:
        return None
    allocation = _allocate_balanced(component_ids, total_budget)
    point_sets = [
        _sample_from_mask(labels == component_id, allocation[component_id], rng)
        for component_id in component_ids
    ]
    return _concat_points(point_sets)


def _top_component_boxes(mask: np.ndarray, top_k: int = 2) -> List[np.ndarray]:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    components = []
    for component_id in range(1, num_labels):
        component_mask = labels == component_id
        area = int(component_mask.sum())
        if area <= 0:
            continue
        box = compute_bbox(component_mask)
        if box is not None:
            components.append((area, box))
    components.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in components[:top_k]]


def _make_prompt(
    strategy: str,
    points: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
    kd_compatible: bool,
    budget_units: int,
    extra_meta: Optional[Dict] = None,
    boxes: Optional[np.ndarray] = None,
) -> Dict:
    if labels is None and points is not None:
        labels = np.ones((len(points),), dtype=np.int32)
    if points is not None:
        points = points.astype(np.float32)
    if labels is not None:
        labels = labels.astype(np.int32)
    if boxes is not None:
        boxes = boxes.astype(np.float32)
    meta = {
        "strategy": strategy,
        "kd_compatible": bool(kd_compatible),
        "sparse_kd_compatible": bool(kd_compatible),
        "oracle_upper_bound": not bool(kd_compatible),
        "prompt_type": "box_point" if (box is not None and points is not None) else ("box" if box is not None else "point"),
        "budget_units": int(budget_units),
        "num_pos_points": int(np.sum(labels == 1)) if labels is not None else 0,
        "num_neg_points": int(np.sum(labels == 0)) if labels is not None else 0,
        "has_box": box is not None or boxes is not None,
        "skip_reason": "",
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "point_coords": points,
        "point_labels": labels,
        "box": None if box is None else box.astype(np.float32),
        "boxes": boxes,
        "meta": meta,
    }


def _make_box_point_prompt(
    strategy: str,
    pos: Optional[np.ndarray],
    neg: Optional[np.ndarray],
    box: Optional[np.ndarray],
    num_prompt_tokens: int,
    box_tokens: int = 2,
    extra_meta: Optional[Dict] = None,
    boxes: Optional[np.ndarray] = None,
) -> Dict:
    points = _concat_points([pos, neg])
    labels = np.concatenate(
        [
            np.ones((0 if pos is None else len(pos),), dtype=np.int32),
            np.zeros((0 if neg is None else len(neg),), dtype=np.int32),
        ]
    )
    budget_units = int(box_tokens) + int(len(labels))
    meta = {
        "prompt_type": "box_point",
        "sparse_kd_compatible": budget_units == int(num_prompt_tokens),
        "kd_compatible": budget_units == int(num_prompt_tokens),
        "oracle_upper_bound": False,
        "budget_units": budget_units,
    }
    if extra_meta:
        meta.update(extra_meta)
    return _make_prompt(
        strategy,
        points,
        labels,
        box,
        budget_units == int(num_prompt_tokens),
        budget_units,
        meta,
        boxes=boxes,
    )


def sample_foreground_prompt(
    strategy: str,
    semantic_mask: np.ndarray,
    num_classes: int,
    foreground_classes: Iterable[int],
    background_id: int,
    ignore_index: int,
    total_budget: int,
    rng: np.random.Generator,
) -> Dict:
    del num_classes
    budget = _strategy_budget(strategy, total_budget)
    classes = _valid_foreground_classes(semantic_mask, foreground_classes, ignore_index)
    foreground_mask = np.isin(semantic_mask, classes) if classes else np.zeros_like(semantic_mask, dtype=bool)
    if not np.any(foreground_mask):
        return _empty_prompt(strategy, strategy not in BOX_STRATEGIES, budget, "no_foreground")

    if strategy.startswith("fg_random_pos_"):
        points = _sample_from_mask(foreground_mask, budget, rng)
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy.startswith("fg_class_balanced_pos75_bgneg25"):
        pos_count = max(1, int(round(budget * 0.75)))
        neg_count = max(0, budget - pos_count)
        allocation = _allocate_balanced(classes, pos_count)
        pos = _sample_class_points(semantic_mask, classes, allocation, rng)
        neg = _sample_negative_near_boundary(
            foreground_mask,
            semantic_mask,
            neg_count,
            rng,
            ignore_index,
            background_id=background_id,
        )
        points = _concat_points([pos, neg])
        labels = np.concatenate(
            [
                np.ones((0 if pos is None else len(pos),), dtype=np.int32),
                np.zeros((0 if neg is None else len(neg),), dtype=np.int32),
            ]
        )
        return _make_prompt(strategy, points, labels, None, True, budget)

    if strategy.startswith("fg_class_balanced_pos"):
        allocation = _allocate_balanced(classes, budget)
        points = _sample_class_points(semantic_mask, classes, allocation, rng)
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy == "fg_area_proportional_pos":
        allocation = _allocate_area(semantic_mask, classes, budget)
        points = _sample_class_points(semantic_mask, classes, allocation, rng)
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy == "fg_connected_component_balanced_pos":
        points = _sample_connected_component_points(foreground_mask, budget, rng)
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy == "fg_distance_center_pos":
        points = _sample_top_distance(foreground_mask, budget, rng)
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy == "fg_mixed_center_boundary_pos":
        center_count = budget // 2
        boundary_count = budget - center_count
        center = _sample_top_distance(foreground_mask, center_count, rng)
        boundary = _sample_boundary_inside(foreground_mask, boundary_count, rng)
        points = _concat_points([center, boundary])
        return _make_prompt(strategy, points, None, None, True, budget)

    if strategy == "fg_pos75_bgneg25":
        pos_count = max(1, int(round(budget * 0.75)))
        neg_count = max(0, budget - pos_count)
        pos = _sample_from_mask(foreground_mask, pos_count, rng)
        neg = _sample_negative_near_boundary(
            foreground_mask,
            semantic_mask,
            neg_count,
            rng,
            ignore_index,
            background_id=background_id,
        )
        points = _concat_points([pos, neg])
        labels = np.concatenate(
            [
                np.ones((0 if pos is None else len(pos),), dtype=np.int32),
                np.zeros((0 if neg is None else len(neg),), dtype=np.int32),
            ]
        )
        return _make_prompt(strategy, points, labels, None, True, budget)

    if strategy == "fg_box":
        box = _bbox_from_mask(foreground_mask)
        return _make_prompt(strategy, None, None, box, False, 4)

    if strategy == "fg_box_points":
        box = _bbox_from_mask(foreground_mask)
        point_count = max(budget - 4, 0)
        points = _sample_from_mask(foreground_mask, point_count, rng)
        return _make_prompt(strategy, points, None, box, False, budget)

    if strategy in {
        "fg_box_pos8_neg6",
        "fg_box_classbalanced_pos8_neg6",
        "fg_box_area_proportional_pos8_neg6",
        "fg_box_center_pos8_neg6",
        "fg_box_mixed_pos8_neg6",
        "fg_expandbox_pos8_neg6",
        "fg_top2box_pos8_neg4",
    }:
        expand_ratio = 0.10 if strategy == "fg_expandbox_pos8_neg6" else 0.0
        box = compute_bbox(foreground_mask, expand_ratio=expand_ratio, image_shape=semantic_mask.shape)
        if box is None:
            return _empty_prompt(strategy, True, budget, "no_box")
        box_tokens = 2
        if strategy == "fg_top2box_pos8_neg4":
            boxes = _top_component_boxes(foreground_mask, top_k=2)
            if len(boxes) >= 2:
                box_tokens = 4
                remaining = max(0, int(total_budget) - box_tokens)
                if remaining <= 1:
                    pos_count, neg_count = max(1, remaining), 0
                else:
                    pos_count = int(round(remaining * (8.0 / 12.0)))
                    pos_count = max(1, min(remaining - 1, pos_count))
                    neg_count = remaining - pos_count
                box = boxes[0]
                boxes_array = np.stack(boxes, axis=0).astype(np.float32)
            else:
                pos_count, neg_count = allocate_pos_neg_budget(total_budget, box_tokens=2)
                boxes_array = None
        else:
            pos_count, neg_count = allocate_pos_neg_budget(total_budget, box_tokens=2)
            boxes_array = None

        if strategy == "fg_box_classbalanced_pos8_neg6":
            allocation = _allocate_balanced(classes, pos_count)
            pos = _sample_class_points(semantic_mask, classes, allocation, rng)
        elif strategy == "fg_box_area_proportional_pos8_neg6":
            allocation = _allocate_area(semantic_mask, classes, pos_count)
            pos = _sample_class_points(semantic_mask, classes, allocation, rng)
        elif strategy == "fg_box_center_pos8_neg6":
            pos = _sample_top_distance(foreground_mask, pos_count, rng)
        elif strategy == "fg_box_mixed_pos8_neg6":
            center_count = pos_count // 2
            boundary_count = pos_count - center_count
            center = _sample_top_distance(foreground_mask, center_count, rng)
            boundary = _sample_boundary_inside(foreground_mask, boundary_count, rng)
            pos = _concat_points([center, boundary])
        else:
            pos = _sample_from_mask(foreground_mask, pos_count, rng)

        neg = sample_negative_in_box(
            box,
            foreground_mask,
            semantic_mask,
            ignore_index,
            neg_count,
            rng,
            prefer_boundary=strategy in {"fg_box_mixed_pos8_neg6", "fg_top2box_pos8_neg4"},
            fallback_global=True,
        )
        return _make_box_point_prompt(
            strategy,
            pos,
            neg,
            box,
            total_budget,
            box_tokens=box_tokens,
            extra_meta={"expand_ratio": expand_ratio},
            boxes=boxes_array,
        )

    raise ValueError(f"Unknown foreground prompt strategy: {strategy}")


def sample_per_class_prompts(
    strategy: str,
    semantic_mask: np.ndarray,
    num_classes: int,
    foreground_classes: Iterable[int],
    background_id: int,
    ignore_index: int,
    total_budget: int,
    rng: np.random.Generator,
) -> Dict[int, Dict]:
    del num_classes
    budget = _strategy_budget(strategy, total_budget)
    classes = _valid_foreground_classes(semantic_mask, foreground_classes, ignore_index)
    prompts = {}
    for class_id in classes:
        target = semantic_mask == class_id
        if strategy == "pc_random_pos_each":
            points = _sample_from_mask(target, budget, rng)
            prompts[class_id] = _make_prompt(strategy, points, None, None, False, budget, {"class_id": class_id})
        elif strategy == "pc_class_pos75_neg25_each":
            pos_count = max(1, int(round(budget * 0.75)))
            neg_count = max(0, budget - pos_count)
            pos = _sample_from_mask(target, pos_count, rng)
            neg = _sample_negative_near_boundary(
                target,
                semantic_mask,
                neg_count,
                rng,
                ignore_index,
                prefer_other_foreground=True,
                background_id=background_id,
            )
            points = _concat_points([pos, neg])
            labels = np.concatenate(
                [
                    np.ones((0 if pos is None else len(pos),), dtype=np.int32),
                    np.zeros((0 if neg is None else len(neg),), dtype=np.int32),
                ]
            )
            prompts[class_id] = _make_prompt(strategy, points, labels, None, False, budget, {"class_id": class_id})
        elif strategy == "pc_distance_center_pos_each":
            points = _sample_top_distance(target, budget, rng)
            prompts[class_id] = _make_prompt(strategy, points, None, None, False, budget, {"class_id": class_id})
        elif strategy == "pc_mixed_center_boundary_pos_each":
            center_count = budget // 2
            boundary_count = budget - center_count
            center = _sample_top_distance(target, center_count, rng)
            boundary = _sample_boundary_inside(target, boundary_count, rng)
            points = _concat_points([center, boundary])
            prompts[class_id] = _make_prompt(strategy, points, None, None, False, budget, {"class_id": class_id})
        elif strategy == "pc_box_each":
            box = _bbox_from_mask(target)
            prompts[class_id] = _make_prompt(strategy, None, None, box, False, 4, {"class_id": class_id})
        elif strategy == "pc_box_points_each":
            box = _bbox_from_mask(target)
            point_count = max(budget - 4, 0)
            points = _sample_from_mask(target, point_count, rng)
            prompts[class_id] = _make_prompt(strategy, points, None, box, False, budget, {"class_id": class_id})
        elif strategy in {
            "pc_box_pos8_neg6_each",
            "pc_box_center_pos8_neg6_each",
            "pc_box_otherfg_neg6_each",
            "pc_expandbox_pos8_neg6_each",
        }:
            expand_ratio = 0.10 if strategy == "pc_expandbox_pos8_neg6_each" else 0.0
            box = compute_bbox(target, expand_ratio=expand_ratio, image_shape=semantic_mask.shape)
            if box is None:
                continue
            pos_count, neg_count = allocate_pos_neg_budget(total_budget, box_tokens=2)
            if strategy == "pc_box_center_pos8_neg6_each":
                pos = _sample_top_distance(target, pos_count, rng)
            else:
                pos = _sample_from_mask(target, pos_count, rng)

            if strategy == "pc_box_otherfg_neg6_each":
                other_fg = (semantic_mask != class_id) & (semantic_mask != background_id) & (semantic_mask != ignore_index)
                neg_parts = []
                remaining_neg = neg_count
                if np.any(other_fg):
                    take = min(int(other_fg.sum()), remaining_neg)
                    other_points = sample_points_from_mask(other_fg, take, rng, replace_if_needed=False)
                    if other_points is not None:
                        neg_parts.append(other_points)
                        remaining_neg -= len(other_points)
                if remaining_neg > 0:
                    box_neg = sample_negative_in_box(
                        box,
                        target | other_fg,
                        semantic_mask,
                        ignore_index,
                        remaining_neg,
                        rng,
                        prefer_boundary=True,
                        fallback_global=True,
                    )
                    neg_parts.append(box_neg)
                neg = _concat_points(neg_parts)
            else:
                neg = sample_negative_in_box(
                    box,
                    target,
                    semantic_mask,
                    ignore_index,
                    neg_count,
                    rng,
                    prefer_boundary=True,
                    fallback_global=True,
                )
            prompts[class_id] = _make_box_point_prompt(
                strategy,
                pos,
                neg,
                box,
                total_budget,
                box_tokens=2,
                extra_meta={"class_id": class_id, "expand_ratio": expand_ratio},
            )
        else:
            raise ValueError(f"Unknown per-class prompt strategy: {strategy}")
    return prompts
