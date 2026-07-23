from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import shlex
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils import LOGGER, nms
from ultralytics.utils.metrics import DetMetrics, box_iou


CLASS_NAMES = {0: "human", 1: "dog"}
TARGET_COCO_CLASSES = {1: 0, 18: 1}
SPLITS = ("train", "val", "test")
DATASET_SCHEMA_VERSION = 9
SCREEN_SCHEMA_VERSION = 4
PERCEPTUAL_MAX_HAMMING_DISTANCE = 8
PERCEPTUAL_MIN_GEOMETRIC_INLIERS = 20
PERCEPTUAL_MIN_INLIER_RATIO = 0.5
RUN_BINDING_SCHEMA_VERSION = 2
RESUME_ARGUMENT_KEYS = (
    "data",
    "epochs",
    "imgsz",
    "batch",
    "patience",
    "device",
    "workers",
    "project",
    "name",
    "pretrained",
    "optimizer",
    "seed",
    "deterministic",
    "rect",
    "val",
    "cache",
    "mosaic",
    "close_mosaic",
    "mixup",
    "copy_paste",
    "degrees",
    "translate",
    "scale",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
)


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class SourceRecord:
    source: str
    source_split: str
    source_id: str
    assigned_split: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[Box, ...]
    kind: str

    @property
    def group_id(self) -> str:
        return f"{self.source}:{self.source_split}:{self.source_id}"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    source: str
    source_split: str
    source_id: str
    group_id: str
    variant: str
    image_path: str
    label_path: str
    width: int
    height: int
    box_count: int
    classes: str


@dataclass(frozen=True)
class PerceptualItem:
    split: str
    key: str
    image_path: Path
    content_hash: str
    difference_hash: int


class _HammingTreeNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, _HammingTreeNode] = {}

    def add(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indices.append(index)
        elif distance in self.children:
            self.children[distance].add(value, index)
        else:
            self.children[distance] = _HammingTreeNode(value, index)

    def query(self, value: int, maximum_distance: int) -> list[int]:
        distance = (self.value ^ value).bit_count()
        matches = list(self.indices) if distance <= maximum_distance else []
        for child_distance, child in self.children.items():
            if distance - maximum_distance <= child_distance <= distance + maximum_distance:
                matches.extend(child.query(value, maximum_distance))
        return matches


def environment_report() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
    }


def stable_int(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def stable_fraction(seed: int, *parts: object) -> float:
    return stable_int(seed, *parts) / 2**64


def stable_take(records: Iterable[SourceRecord], limit: int | None, seed: int, salt: str) -> list[SourceRecord]:
    ordered = sorted(records, key=lambda record: stable_int(seed, salt, record.group_id))
    return ordered if limit is None else ordered[:limit]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] > resized[:, :-1]).reshape(-1)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _square_letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    image_size: int,
    scaleup: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    ratio = min(image_size / width, image_size / height)
    if not scaleup:
        ratio = min(ratio, 1.0)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    horizontal_padding = (image_size - resized_width) / 2.0
    vertical_padding = (image_size - resized_height) / 2.0
    left = round(horizontal_padding - 0.1)
    top = round(vertical_padding - 0.1)
    frame = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    frame[top : top + resized_height, left : left + resized_width] = image

    transformed = boxes.copy()
    if len(transformed):
        transformed[:, [0, 2]] = transformed[:, [0, 2]] * ratio + left
        transformed[:, [1, 3]] = transformed[:, [1, 3]] * ratio + top
    return frame, transformed


def _pytorch_detection_network(
    model: YOLO,
    device: str,
) -> tuple[torch.nn.Module, torch.device]:
    network = model.model
    if not isinstance(network, torch.nn.Module):
        raise TypeError("Multi-label evaluation requires a PyTorch model.")
    inference_device = torch.device(device)
    return network.to(inference_device).eval(), inference_device


def _multi_label_predictions(
    network: torch.nn.Module,
    frames: Sequence[np.ndarray],
    inference_device: torch.device,
    class_count: int,
    confidence: float,
    iou: float,
    max_detections: int,
    classes: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    image_tensor = torch.from_numpy(
        np.stack([frame[:, :, ::-1] for frame in frames])
    ).permute(0, 3, 1, 2)
    image_tensor = image_tensor.contiguous().to(inference_device).float() / 255.0
    with torch.inference_mode():
        raw_predictions = network(image_tensor)
        return nms.non_max_suppression(
            raw_predictions,
            conf_thres=confidence,
            iou_thres=iou,
            classes=classes,
            multi_label=True,
            max_det=max_detections,
            nc=class_count,
            end2end=getattr(network, "end2end", False),
        )


def _clamp_box(
    cls: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    min_box_pixels: float,
) -> Box | None:
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), 0.0), float(width))
    y2 = min(max(float(y2), 0.0), float(height))
    box = Box(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2)
    return box if box.width >= min_box_pixels and box.height >= min_box_pixels else None


def _record_kind(boxes: Sequence[Box]) -> str:
    classes = {box.cls for box in boxes}
    if classes == {0, 1}:
        return "mixed"
    if classes == {0}:
        return "person"
    if classes == {1}:
        return "dog"
    return "negative"


def _box_iou_value(left: Box, right: Box) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def _merge_duplicate_dog_boxes(records: Sequence[SourceRecord]) -> tuple[Box, ...]:
    merged = list(records[0].boxes)
    for record in records[1:]:
        for box in record.boxes:
            overlaps = [_box_iou_value(box, candidate) for candidate in merged]
            best_index = int(np.argmax(overlaps)) if overlaps else -1
            if best_index >= 0 and overlaps[best_index] >= 0.8:
                existing = merged[best_index]
                merged[best_index] = Box(
                    cls=1,
                    x1=min(existing.x1, box.x1),
                    y1=min(existing.y1, box.y1),
                    x2=max(existing.x2, box.x2),
                    y2=max(existing.y2, box.y2),
                )
            else:
                merged.append(box)
    return tuple(sorted(merged, key=lambda box: (box.x1, box.y1, box.x2, box.y2)))


def load_coco_records(
    image_dir: Path,
    annotation_path: Path,
    source_split: str,
    seed: int,
    min_box_pixels: float,
) -> tuple[list[SourceRecord], dict[str, int]]:
    with annotation_path.open() as handle:
        coco = json.load(handle)

    target_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in TARGET_COCO_CLASSES:
            target_annotations[annotation["image_id"]].append(annotation)

    records: list[SourceRecord] = []
    stats = Counter()
    for image in coco["images"]:
        annotations = target_annotations[image["id"]]
        if any(annotation.get("iscrowd", 0) for annotation in annotations):
            stats["excluded_target_crowd"] += 1
            continue

        boxes: list[Box] = []
        invalid_target = False
        for annotation in annotations:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            box = _clamp_box(
                TARGET_COCO_CLASSES[annotation["category_id"]],
                x,
                y,
                x + width,
                y + height,
                int(image["width"]),
                int(image["height"]),
                min_box_pixels,
            )
            if box is None:
                invalid_target = True
                break
            boxes.append(box)

        if invalid_target:
            stats["excluded_invalid_target_box"] += 1
            continue

        if source_split == "train2017":
            assigned_split = "train"
        else:
            assigned_split = (
                "val" if stable_fraction(seed, "coco-val-test", image["id"]) < 0.5 else "test"
            )

        image_path = image_dir / image["file_name"]
        record = SourceRecord(
            source="coco",
            source_split=source_split,
            source_id=str(image["id"]),
            assigned_split=assigned_split,
            image_path=image_path,
            width=int(image["width"]),
            height=int(image["height"]),
            boxes=tuple(boxes),
            kind=_record_kind(boxes),
        )
        records.append(record)
        stats[f"candidate_{assigned_split}_{record.kind}"] += 1

    stats["records"] = len(records)
    return records, dict(stats)


def load_stanford_records(
    image_root: Path,
    annotation_root: Path,
    seed: int,
    min_box_pixels: float,
) -> tuple[list[SourceRecord], dict[str, int]]:
    records_by_digest: dict[str, list[SourceRecord]] = defaultdict(list)
    stats = Counter()
    annotation_paths = sorted(path for path in annotation_root.rglob("*") if path.is_file())

    for annotation_path in annotation_paths:
        root = ET.parse(annotation_path).getroot()
        size = root.find("size")
        if size is None:
            stats["excluded_missing_size"] += 1
            continue
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            stats["excluded_invalid_size"] += 1
            continue

        boxes: list[Box] = []
        invalid_target = False
        for obj in root.findall("object"):
            bounds = obj.find("bndbox")
            if bounds is None:
                invalid_target = True
                break
            # Stanford Dogs stores zero-based, inclusive box coordinates.
            box = _clamp_box(
                1,
                float(bounds.findtext("xmin", "nan")),
                float(bounds.findtext("ymin", "nan")),
                float(bounds.findtext("xmax", "nan")) + 1.0,
                float(bounds.findtext("ymax", "nan")) + 1.0,
                width,
                height,
                min_box_pixels,
            )
            if box is None:
                invalid_target = True
                break
            boxes.append(box)

        if invalid_target or not boxes:
            stats["excluded_invalid_target_box"] += 1
            continue

        relative_path = annotation_path.relative_to(annotation_root)
        source_id = relative_path.as_posix()
        image_path = image_root / relative_path.parent / f"{relative_path.name}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image_digest = sha256_file(image_path)
        split_value = stable_fraction(seed, "stanford-split", image_digest)
        assigned_split = "train" if split_value < 0.8 else "val" if split_value < 0.9 else "test"
        record = SourceRecord(
            source="stanford",
            source_split="stanford-dogs",
            source_id=source_id,
            assigned_split=assigned_split,
            image_path=image_path,
            width=width,
            height=height,
            boxes=tuple(boxes),
            kind="dog",
        )
        records_by_digest[image_digest].append(record)

    records: list[SourceRecord] = []
    for duplicate_records in records_by_digest.values():
        ordered = sorted(duplicate_records, key=lambda record: record.source_id)
        canonical = ordered[0]
        if any(
            (record.width, record.height) != (canonical.width, canonical.height)
            for record in ordered[1:]
        ):
            raise ValueError(f"Duplicate Stanford image has conflicting dimensions: {canonical.image_path}")
        merged_boxes = _merge_duplicate_dog_boxes(ordered)
        records.append(
            SourceRecord(
                source=canonical.source,
                source_split=canonical.source_split,
                source_id=canonical.source_id,
                assigned_split=canonical.assigned_split,
                image_path=canonical.image_path,
                width=canonical.width,
                height=canonical.height,
                boxes=merged_boxes,
                kind="dog",
            )
        )
        stats[f"candidate_{canonical.assigned_split}_dog"] += 1
        stats["objects"] += len(merged_boxes)
        if len(ordered) > 1:
            stats["exact_duplicate_groups"] += 1
            stats["exact_duplicate_records_removed"] += len(ordered) - 1

    records.sort(key=lambda record: record.source_id)
    stats["records"] = len(records)
    return records, dict(stats)


def screen_stanford_person_candidates(
    records: Sequence[SourceRecord],
    cache_path: Path,
    model_path: str | Path,
    device: str,
    confidence: float = 0.2,
    image_size: int = 640,
    batch: int = 16,
    rebuild: bool = False,
) -> set[str]:
    model = YOLO(str(model_path))
    model_file = Path(model.ckpt_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Unable to resolve screening model: {model.ckpt_path}")
    source_digest = source_image_digest(records, "Stanford screening")
    model_identity = sha256_file(model_file)
    screen_config = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "model": str(model_file),
        "model_identity": model_identity,
        "confidence": confidence,
        "image_size": image_size,
        "batch": batch,
        "rect": False,
        "scaleup": True,
        "multi_label": True,
        "device": device,
        "ultralytics": ultralytics.__version__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(screen_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if cache_path.exists() and not rebuild:
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{cache_path} was produced by a different screening configuration. "
                "Set REBUILD_STANFORD_SCREEN=True to replace it."
            )
        exclusions = _validated_screen_exclusions(
            cached,
            {record.source_id for record in records},
        )
        print(f"Reusing Stanford person-screen cache: {cache_path}")
        return exclusions

    if batch < 1:
        raise ValueError("batch must be at least 1.")
    network, inference_device = _pytorch_detection_network(model, device)
    candidates = []
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    for start in range(0, len(records), batch):
        record_chunk = records[start : start + batch]
        frames = []
        for record in record_chunk:
            image = cv2.imread(str(record.image_path))
            if image is None:
                raise ValueError(f"Unreadable screening image: {record.image_path}")
            frame, _ = _square_letterbox(
                image,
                empty_boxes,
                image_size,
                scaleup=True,
            )
            frames.append(frame)
        predictions = _multi_label_predictions(
            network,
            frames,
            inference_device,
            class_count=len(model.names),
            confidence=confidence,
            iou=0.7,
            max_detections=20,
            classes=[0],
        )
        for record, prediction in zip(record_chunk, predictions, strict=True):
            if len(prediction):
                candidates.append(
                    {
                        "source_id": record.source_id,
                        "detections": len(prediction),
                        "max_confidence": float(prediction[:, 4].max().item()),
                    }
                )

    candidates.sort(key=lambda candidate: candidate["source_id"])
    excluded_source_ids = [candidate["source_id"] for candidate in candidates]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "config": screen_config,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "excluded_source_ids": excluded_source_ids,
        "result_digest": _screen_result_digest(candidates, excluded_source_ids),
    }
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return set(excluded_source_ids)


def select_sources(
    coco_records: Sequence[SourceRecord],
    stanford_records: Sequence[SourceRecord],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[SourceRecord]], dict[str, dict[str, int]]]:
    seed = int(config["seed"])
    person_limits = config["max_coco_person_only"]
    stanford_limits = config["max_stanford"]
    negative_fraction = float(config["negative_fraction"])
    if not 0.0 <= negative_fraction < 1.0:
        raise ValueError("negative_fraction must be in [0, 1).")

    selected: dict[str, list[SourceRecord]] = {}
    reports: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        coco_split = [record for record in coco_records if record.assigned_split == split]
        stanford_split = [record for record in stanford_records if record.assigned_split == split]

        person_only = stable_take(
            (record for record in coco_split if record.kind == "person"),
            person_limits[split],
            seed,
            f"{split}-coco-person",
        )
        coco_dog_or_mixed = stable_take(
            (record for record in coco_split if record.kind in {"dog", "mixed"}),
            None,
            seed,
            f"{split}-coco-dog",
        )
        stanford_dogs = stable_take(
            stanford_split,
            stanford_limits[split],
            seed,
            f"{split}-stanford",
        )
        positives = [*person_only, *coco_dog_or_mixed, *stanford_dogs]

        negative_target = round(len(positives) * negative_fraction / (1.0 - negative_fraction))
        negatives = stable_take(
            (record for record in coco_split if record.kind == "negative"),
            negative_target,
            seed,
            f"{split}-negative",
        )

        split_records = sorted(
            [*positives, *negatives],
            key=lambda record: stable_int(seed, f"{split}-final", record.group_id),
        )
        selected[split] = split_records
        reports[split] = {
            "coco_person_only": len(person_only),
            "coco_dog_or_mixed": len(coco_dog_or_mixed),
            "stanford_dog": len(stanford_dogs),
            "negative": len(negatives),
            "total": len(split_records),
        }

    assert_source_disjoint(selected)
    return selected, reports


def assert_source_disjoint(selected: Mapping[str, Sequence[SourceRecord]]) -> None:
    groups_by_split = {
        split: {record.group_id for record in records} for split, records in selected.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(f"Source leakage between {left} and {right}: {examples}")


def _selection_counts(records: Sequence[SourceRecord]) -> dict[str, int]:
    counts = {
        "coco_person_only": sum(
            record.source == "coco" and record.kind == "person" for record in records
        ),
        "coco_dog_or_mixed": sum(
            record.source == "coco" and record.kind in {"dog", "mixed"} for record in records
        ),
        "stanford_dog": sum(record.source == "stanford" for record in records),
        "negative": sum(record.kind == "negative" for record in records),
    }
    counts["total"] = len(records)
    return counts


def _geometrically_matching_images(
    left: PerceptualItem,
    right: PerceptualItem,
    detector: cv2.ORB,
    matcher: cv2.BFMatcher,
    feature_cache: dict[Path, tuple[Sequence[cv2.KeyPoint], np.ndarray | None]],
) -> bool:
    def features(path: Path) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
        if path not in feature_cache:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unreadable source image: {path}")
            feature_cache[path] = detector.detectAndCompute(image, None)
        return feature_cache[path]

    left_points, left_descriptors = features(left.image_path)
    right_points, right_descriptors = features(right.image_path)
    if left_descriptors is None or right_descriptors is None:
        return False
    matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < PERCEPTUAL_MIN_GEOMETRIC_INLIERS:
        return False

    left_coordinates = np.float32(
        [left_points[match.queryIdx].pt for match in good_matches]
    )
    right_coordinates = np.float32(
        [right_points[match.trainIdx].pt for match in good_matches]
    )
    _, inlier_mask = cv2.findHomography(
        left_coordinates,
        right_coordinates,
        cv2.RANSAC,
        5.0,
    )
    if inlier_mask is None:
        return False
    inliers = int(inlier_mask.sum())
    return (
        inliers >= PERCEPTUAL_MIN_GEOMETRIC_INLIERS
        and inliers / len(good_matches) >= PERCEPTUAL_MIN_INLIER_RATIO
    )


def perceptual_duplicate_clusters(
    items: Sequence[PerceptualItem],
) -> tuple[list[list[int]], int]:
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    tree: _HammingTreeNode | None = None
    detector = cv2.ORB_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    feature_cache: dict[Path, tuple[Sequence[cv2.KeyPoint], np.ndarray | None]] = {}
    verified_near_pairs = 0

    for index, item in enumerate(items):
        if tree is not None:
            for candidate_index in tree.query(
                item.difference_hash,
                PERCEPTUAL_MAX_HAMMING_DISTANCE,
            ):
                candidate = items[candidate_index]
                distance = (
                    item.difference_hash ^ candidate.difference_hash
                ).bit_count()
                if (
                    item.content_hash == candidate.content_hash
                    or _geometrically_matching_images(
                        item,
                        candidate,
                        detector,
                        matcher,
                        feature_cache,
                    )
                ):
                    union(index, candidate_index)
                    verified_near_pairs += item.content_hash != candidate.content_hash
        if tree is None:
            tree = _HammingTreeNode(item.difference_hash, index)
        else:
            tree.add(item.difference_hash, index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        grouped[find(index)].append(index)
    return [indices for indices in grouped.values() if len(indices) > 1], verified_near_pairs


def remove_perceptual_source_duplicates(
    selected: Mapping[str, Sequence[SourceRecord]],
) -> tuple[dict[str, list[SourceRecord]], dict[str, Any]]:
    items: list[PerceptualItem] = []
    record_by_group: dict[str, SourceRecord] = {}
    total = sum(len(records) for records in selected.values())
    processed = 0
    print(f"Checking {total} selected sources for perceptual duplicates")
    for split in SPLITS:
        for record in selected[split]:
            image = cv2.imread(str(record.image_path))
            if image is None:
                raise ValueError(f"Unreadable source image: {record.image_path}")
            items.append(
                PerceptualItem(
                    split=split,
                    key=record.group_id,
                    image_path=record.image_path,
                    content_hash=sha256_file(record.image_path),
                    difference_hash=_difference_hash(image),
                )
            )
            record_by_group[record.group_id] = record
            processed += 1
            if processed % 1000 == 0:
                print(f"  perceptual hashes: {processed}/{total}")

    clusters, verified_near_pairs = perceptual_duplicate_clusters(items)
    split_priority = {split: index for index, split in enumerate(SPLITS)}
    keep_groups = set(record_by_group)
    cross_split_groups = 0
    removed_by_split = Counter()
    for cluster in clusters:
        entries = [items[index] for index in cluster]
        cross_split_groups += len({item.split for item in entries}) > 1
        best_split_priority = min(split_priority[item.split] for item in entries)
        candidates = [
            record_by_group[item.key]
            for item in entries
            if split_priority[item.split] == best_split_priority
        ]
        canonical = min(
            candidates,
            key=lambda record: (
                record.source != "coco",
                -len(record.boxes),
                -(record.width * record.height),
                record.group_id,
            ),
        )
        for item in entries:
            if item.key != canonical.group_id:
                keep_groups.remove(item.key)
                removed_by_split[item.split] += 1

    deduplicated = {
        split: [record for record in selected[split] if record.group_id in keep_groups]
        for split in SPLITS
    }
    assert_source_disjoint(deduplicated)
    return deduplicated, {
        "groups": len(clusters),
        "cross_split_groups": cross_split_groups,
        "verified_near_duplicate_pairs": verified_near_pairs,
        "records_removed": sum(removed_by_split.values()),
        "removed_by_split": dict(removed_by_split),
    }


def selection_digest(selected: Mapping[str, Sequence[SourceRecord]]) -> str:
    rows = []
    for split in SPLITS:
        for record in sorted(selected[split], key=lambda item: item.group_id):
            boxes = ";".join(
                f"{box.cls},{box.x1:.8f},{box.y1:.8f},{box.x2:.8f},{box.y2:.8f}"
                for box in record.boxes
            )
            rows.append(
                f"{split}|{record.group_id}|{record.width}|{record.height}|{record.kind}|{boxes}"
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def source_image_digest(records: Sequence[SourceRecord], label: str) -> str:
    digest = hashlib.sha256()
    ordered = sorted(records, key=lambda record: record.group_id)
    print(f"Hashing {len(ordered)} {label} source images")
    for index, record in enumerate(ordered, start=1):
        if index % 1000 == 0:
            print(f"  {label}: {index}/{len(ordered)}")
        if not record.image_path.exists():
            raise FileNotFoundError(record.image_path)
        digest.update(record.group_id.encode("utf-8"))
        digest.update(sha256_file(record.image_path).encode("ascii"))
    return digest.hexdigest()


def _screen_result_digest(
    candidates: Sequence[Mapping[str, Any]],
    excluded_source_ids: Sequence[str],
) -> str:
    payload = {
        "candidates": list(candidates),
        "excluded_source_ids": list(excluded_source_ids),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_screen_exclusions(
    cached: Mapping[str, Any],
    valid_source_ids: set[str],
) -> set[str]:
    candidates = cached.get("candidates")
    excluded = cached.get("excluded_source_ids")
    if not isinstance(candidates, list) or not isinstance(excluded, list):
        raise RuntimeError("Stanford screening cache has an invalid result payload.")
    if cached.get("candidate_count") != len(candidates):
        raise RuntimeError("Stanford screening cache candidate count is inconsistent.")

    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("Stanford screening cache contains an invalid candidate.")
        source_id = candidate.get("source_id")
        detections = candidate.get("detections")
        confidence = candidate.get("max_confidence")
        if (
            not isinstance(source_id, str)
            or source_id not in valid_source_ids
            or not isinstance(detections, int)
            or detections < 1
            or not isinstance(confidence, (int, float))
            or not np.isfinite(confidence)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise RuntimeError("Stanford screening cache contains invalid candidate values.")
        candidate_ids.append(source_id)

    if (
        len(candidate_ids) != len(set(candidate_ids))
        or excluded != candidate_ids
        or any(not isinstance(source_id, str) for source_id in excluded)
    ):
        raise RuntimeError("Stanford screening cache exclusions are inconsistent.")
    expected_digest = _screen_result_digest(candidates, excluded)
    if cached.get("result_digest") != expected_digest:
        raise RuntimeError("Stanford screening cache result digest is invalid.")
    return set(excluded)


def config_fingerprint(
    config: Mapping[str, Any],
    selected: Mapping[str, Sequence[SourceRecord]],
    source_digest: str | None = None,
) -> str:
    if source_digest is None:
        source_digest = source_image_digest(
            [record for split in SPLITS for record in selected[split]],
            "dataset",
        )
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config": config,
        "selection_digest": selection_digest(selected),
        "source_image_digest": source_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_stem(record: SourceRecord) -> str:
    if record.source == "coco":
        return f"coco_{record.source_split}_{int(record.source_id):012d}"
    digest = hashlib.sha256(record.source_id.encode("utf-8")).hexdigest()[:10]
    return f"stanford_{digest}_{Path(record.source_id).name}"


def _link_or_copy(source: Path, destination: Path) -> str:
    shutil.copy2(source, destination)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        with destination.open("rb+") as handle:
            is_jpeg = handle.read(2) == b"\xff\xd8"
            handle.seek(-2, os.SEEK_END)
            if is_jpeg and handle.read() != b"\xff\xd9":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\xff\xd9")
                return "copy_jpeg_eoi_repaired"
    return "copy"


def _write_yolo_labels(path: Path, boxes: Sequence[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        x_center = ((box.x1 + box.x2) / 2.0) / width
        y_center = ((box.y1 + box.y2) / 2.0) / height
        box_width = box.width / width
        box_height = box.height / height
        values = (x_center, y_center, box_width, box_height)
        if not all(np.isfinite(values)) or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid normalized box for {path}: {values}")
        lines.append(f"{box.cls} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _transform_boxes_to_crop(
    boxes: Sequence[Box],
    crop_x1: int,
    crop_y1: int,
    side: int,
    min_box_pixels: float,
) -> list[Box] | None:
    transformed = []
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side
    for box in boxes:
        x1 = max(box.x1, crop_x1)
        y1 = max(box.y1, crop_y1)
        x2 = min(box.x2, crop_x2)
        y2 = min(box.y2, crop_y2)
        transformed_box = _clamp_box(
            box.cls,
            x1 - crop_x1,
            y1 - crop_y1,
            x2 - crop_x1,
            y2 - crop_y1,
            side,
            side,
            min_box_pixels,
        )
        intersects_crop = x2 > x1 and y2 > y1
        if intersects_crop and transformed_box is None:
            return None
        if transformed_box is not None:
            transformed.append(transformed_box)
    return transformed


def create_partial_view(
    image: np.ndarray,
    boxes: Sequence[Box],
    seed: int,
    group_id: str,
    min_box_pixels: float,
    min_visibility: float,
) -> tuple[np.ndarray, list[Box]] | None:
    if not boxes:
        return None

    height, width = image.shape[:2]
    rng = random.Random(stable_int(seed, "partial-view", group_id))
    anchor = boxes[rng.randrange(len(boxes))]
    visible_fraction = rng.uniform(0.4, 0.75) if anchor.cls == 0 else rng.uniform(0.6, 0.9)
    visible_height = anchor.height * visible_fraction
    min_dimension = min(width, height)
    side = int(
        max(
            rng.uniform(0.55, 0.85) * min_dimension,
            anchor.width * 1.4,
            visible_height * 1.2,
            64,
        )
    )
    side = min(side, min_dimension)
    if side < 64:
        return None

    anchor_center_x = (anchor.x1 + anchor.x2) / 2.0
    center_x = anchor_center_x + rng.uniform(-0.12, 0.12) * side
    crop_x1 = int(round(center_x - side / 2.0))
    crop_x1 = min(max(crop_x1, 0), width - side)

    desired_visible_top = anchor.y2 - visible_height
    crop_y1 = int(round(desired_visible_top - rng.uniform(0.03, 0.12) * side))
    crop_y1 = min(max(crop_y1, 0), height - side)

    anchor_intersection_width = max(
        0.0,
        min(anchor.x2, crop_x1 + side) - max(anchor.x1, crop_x1),
    )
    anchor_intersection_height = max(
        0.0,
        min(anchor.y2, crop_y1 + side) - max(anchor.y1, crop_y1),
    )
    anchor_visibility = (
        anchor_intersection_width * anchor_intersection_height / anchor.area if anchor.area else 0.0
    )
    if anchor_visibility < min_visibility:
        return None

    transformed = _transform_boxes_to_crop(
        boxes,
        crop_x1,
        crop_y1,
        side,
        min_box_pixels,
    )
    if transformed is None or not transformed:
        return None

    crop = image[crop_y1 : crop_y1 + side, crop_x1 : crop_x1 + side]
    if crop.shape[:2] != (side, side):
        raise ValueError(f"Unexpected partial crop shape for {group_id}: {crop.shape}")
    return crop, transformed


def _sample_record(
    sample_id: str,
    record: SourceRecord,
    split: str,
    variant: str,
    image_path: Path,
    label_path: Path,
    width: int,
    height: int,
    boxes: Sequence[Box],
    dataset_root: Path,
) -> SampleRecord:
    classes = ",".join(str(value) for value in sorted({box.cls for box in boxes}))
    return SampleRecord(
        sample_id=sample_id,
        split=split,
        source=record.source,
        source_split=record.source_split,
        source_id=record.source_id,
        group_id=record.group_id,
        variant=variant,
        image_path=image_path.relative_to(dataset_root).as_posix(),
        label_path=label_path.relative_to(dataset_root).as_posix(),
        width=width,
        height=height,
        box_count=len(boxes),
        classes=classes,
    )


def _write_manifest(path: Path, samples: Sequence[SampleRecord]) -> None:
    fieldnames = list(SampleRecord.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({field: getattr(sample, field) for field in fieldnames})


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_source_manifest(
    path: Path,
    selected: Mapping[str, Sequence[SourceRecord]],
) -> None:
    fieldnames = [
        "split",
        "source",
        "source_split",
        "source_id",
        "group_id",
        "image_path",
        "width",
        "height",
        "kind",
        "box_count",
        "classes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLITS:
            for record in selected[split]:
                writer.writerow(
                    {
                        "split": split,
                        "source": record.source,
                        "source_split": record.source_split,
                        "source_id": record.source_id,
                        "group_id": record.group_id,
                        "image_path": str(record.image_path.resolve()),
                        "width": record.width,
                        "height": record.height,
                        "kind": record.kind,
                        "box_count": len(record.boxes),
                        "classes": ",".join(str(value) for value in sorted({box.cls for box in record.boxes})),
                    }
                )


def _write_list_files(dataset_root: Path, samples: Sequence[SampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_samples = [sample for sample in samples if sample.split == split]
        variants = {
            split: split_samples,
            f"{split}_coco": [sample for sample in split_samples if sample.source == "coco"],
            f"{split}_stanford": [sample for sample in split_samples if sample.source == "stanford"],
        }
        for name, selected_samples in variants.items():
            path = dataset_root / f"{name}.txt"
            lines = [f"./{sample.image_path}" for sample in selected_samples]
            path.write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[name] = len(lines)
    return counts


def _write_dataset_yaml(path: Path, dataset_root: Path, val_list: str, test_list: str) -> None:
    content = {
        "path": str(dataset_root.resolve()),
        "train": "train.txt",
        "val": val_list,
        "test": test_list,
        "names": CLASS_NAMES,
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))


def _dataset_yaml_paths(dataset_root: Path, yaml_dir: Path) -> dict[str, Path]:
    dataset_name = dataset_root.name
    return {
        "combined": yaml_dir / f"{dataset_name}.yaml",
        "coco": yaml_dir / f"{dataset_name}_coco.yaml",
        "stanford": yaml_dir / f"{dataset_name}_stanford.yaml",
    }


def write_dataset_yamls(dataset_root: Path, yaml_dir: Path) -> dict[str, Path]:
    yaml_dir.mkdir(parents=True, exist_ok=True)
    paths = _dataset_yaml_paths(dataset_root, yaml_dir)
    _write_dataset_yaml(paths["combined"], dataset_root, "val.txt", "test.txt")
    _write_dataset_yaml(paths["coco"], dataset_root, "val_coco.txt", "test_coco.txt")
    _write_dataset_yaml(paths["stanford"], dataset_root, "val_stanford.txt", "test_stanford.txt")
    return paths


def clear_ultralytics_label_caches(dataset_root: Path) -> int:
    cache_paths = sorted((dataset_root / "labels").rglob("*.cache"))
    for cache_path in cache_paths:
        cache_path.unlink()
    if cache_paths:
        print(f"Removed {len(cache_paths)} Ultralytics label cache file(s).")
    return len(cache_paths)


def load_source_records(
    config: Mapping[str, Any],
) -> tuple[list[SourceRecord], list[SourceRecord], dict[str, Any]]:
    coco_root = Path(config["coco_root"])
    stanford_root = Path(config["stanford_root"])
    seed = int(config["seed"])
    min_box_pixels = float(config["min_box_pixels"])

    coco_train, coco_train_stats = load_coco_records(
        coco_root / "train2017",
        coco_root / "annotations/instances_train2017.json",
        "train2017",
        seed,
        min_box_pixels,
    )
    coco_val, coco_val_stats = load_coco_records(
        coco_root / "val2017",
        coco_root / "annotations/instances_val2017.json",
        "val2017",
        seed,
        min_box_pixels,
    )
    stanford, stanford_stats = load_stanford_records(
        stanford_root / "images/Images",
        stanford_root / "annotations/Annotation",
        seed,
        min_box_pixels,
    )
    report = {
        "coco_train": coco_train_stats,
        "coco_val": coco_val_stats,
        "stanford": stanford_stats,
    }
    return [*coco_train, *coco_val], stanford, report


def select_loaded_sources(
    coco_records: Sequence[SourceRecord],
    stanford_records: Sequence[SourceRecord],
    config: Mapping[str, Any],
    stanford_exclusions: set[str] | None = None,
) -> tuple[dict[str, list[SourceRecord]], dict[str, Any]]:
    exclusions = stanford_exclusions or set()
    filtered_stanford = [record for record in stanford_records if record.source_id not in exclusions]
    selected, selection_before_deduplication = select_sources(
        coco_records,
        filtered_stanford,
        config,
    )
    selected, perceptual_report = remove_perceptual_source_duplicates(selected)
    selection_stats = {
        split: _selection_counts(selected[split])
        for split in SPLITS
    }
    report = {
        "selection": selection_stats,
        "selection_before_perceptual_deduplication": selection_before_deduplication,
        "perceptual_deduplication": perceptual_report,
        "selection_digest": selection_digest(selected),
        "stanford_person_screen_exclusions": len(exclusions),
        "stanford_after_person_screen": len(filtered_stanford),
    }
    return selected, report


def load_and_select_sources(config: Mapping[str, Any]) -> tuple[
    dict[str, list[SourceRecord]],
    dict[str, Any],
]:
    coco_records, stanford_records, load_report = load_source_records(config)
    selected, selection_report = select_loaded_sources(
        coco_records,
        stanford_records,
        config,
    )
    return selected, {**load_report, **selection_report}


def build_dataset(
    config: Mapping[str, Any],
    selected: Mapping[str, Sequence[SourceRecord]],
    source_report: Mapping[str, Any],
    rebuild: bool,
    build_if_missing: bool = True,
) -> dict[str, Any]:
    final_root = Path(config["dataset_root"])
    yaml_dir = Path(config["yaml_dir"])
    selected_records = [record for split in SPLITS for record in selected[split]]
    source_digest = source_image_digest(selected_records, "dataset fingerprint")
    fingerprint = config_fingerprint(config, selected, source_digest)
    metadata_path = final_root / "dataset_metadata.json"

    if final_root.exists() and not rebuild:
        if not metadata_path.exists():
            raise RuntimeError(
                f"{final_root} exists without metadata. Set REBUILD_DATASET=True to replace it."
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{final_root} was built with different sources or configuration. "
                "Use a new dataset name or set REBUILD_DATASET=True."
            )
        yaml_paths = write_dataset_yamls(final_root, yaml_dir)
        serialized_yaml_paths = {name: str(path.resolve()) for name, path in yaml_paths.items()}
        if metadata.get("yaml_paths") != serialized_yaml_paths:
            raise RuntimeError(f"{metadata_path} contains inconsistent YAML paths.")
        required = [
            final_root / "manifest.csv",
            final_root / "source_manifest.csv",
            final_root / "audit_report.json",
            *(final_root / f"{split}{suffix}.txt" for split in SPLITS for suffix in ("", "_coco", "_stanford")),
            *yaml_paths.values(),
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f"{final_root} is incomplete. Set REBUILD_DATASET=True.")
        clear_ultralytics_label_caches(final_root)
        audit_report = audit_dataset(final_root, write_report=False)
        expected_artifact_digest = metadata.get("audit", {}).get("artifact_digest")
        if not expected_artifact_digest or audit_report["artifact_digest"] != expected_artifact_digest:
            raise RuntimeError(
                f"{final_root} artifacts differ from the verified build. "
                "Set REBUILD_DATASET=True to replace them."
            )
        (final_root / "audit_report.json").write_text(
            json.dumps(audit_report, indent=2, sort_keys=True)
        )
        print(f"Reusing explicitly verified dataset: {final_root}")
        return metadata

    if not build_if_missing:
        raise FileNotFoundError(
            f"No verified dataset exists at {final_root}. Set BUILD_DATASET=True to create it."
        )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        prefix=f".{final_root.name}.staging-",
        dir=final_root.parent,
        ignore_cleanup_errors=True,
    )
    dataset_root = Path(staging.name)
    for split in SPLITS:
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    samples: list[SampleRecord] = []
    materialization = Counter()
    partial_fraction = float(config["partial_view_fraction"])
    seed = int(config["seed"])
    min_box_pixels = float(config["min_box_pixels"])
    min_visibility = float(config["min_partial_visibility"])

    for split in SPLITS:
        print(f"Building {split}: {len(selected[split])} source images")
        for index, record in enumerate(selected[split], start=1):
            if index % 1000 == 0:
                print(f"  {split}: {index}/{len(selected[split])}")
            if not record.image_path.exists():
                raise FileNotFoundError(f"Missing source image: {record.image_path}")
            image = cv2.imread(str(record.image_path))
            if image is None:
                raise ValueError(f"Unreadable source image: {record.image_path}")
            actual_height, actual_width = image.shape[:2]
            if (actual_width, actual_height) != (record.width, record.height):
                raise ValueError(
                    f"Annotation/image size mismatch for {record.image_path}: "
                    f"annotation={(record.width, record.height)}, image={(actual_width, actual_height)}"
                )

            stem = _safe_stem(record)
            image_path = dataset_root / "images" / split / f"{stem}.jpg"
            label_path = dataset_root / "labels" / split / f"{stem}.txt"
            materialization[_link_or_copy(record.image_path, image_path)] += 1
            _write_yolo_labels(label_path, record.boxes, record.width, record.height)
            samples.append(
                _sample_record(
                    stem,
                    record,
                    split,
                    "full",
                    image_path,
                    label_path,
                    record.width,
                    record.height,
                    record.boxes,
                    dataset_root,
                )
            )

            create_partial = (
                split == "train"
                and bool(record.boxes)
                and stable_fraction(seed, "partial-select", record.group_id) < partial_fraction
            )
            if not create_partial:
                continue

            partial = create_partial_view(
                image,
                record.boxes,
                seed,
                record.group_id,
                min_box_pixels,
                min_visibility,
            )
            if partial is None:
                materialization["partial_skipped"] += 1
                continue
            crop, crop_boxes = partial
            partial_stem = f"{stem}_partial"
            partial_image_path = dataset_root / "images" / split / f"{partial_stem}.jpg"
            partial_label_path = dataset_root / "labels" / split / f"{partial_stem}.txt"
            if not cv2.imwrite(
                str(partial_image_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            ):
                raise OSError(f"Failed to write {partial_image_path}")
            side = crop.shape[0]
            _write_yolo_labels(partial_label_path, crop_boxes, side, side)
            samples.append(
                _sample_record(
                    partial_stem,
                    record,
                    split,
                    "partial",
                    partial_image_path,
                    partial_label_path,
                    side,
                    side,
                    crop_boxes,
                    dataset_root,
                )
            )
            materialization["partial"] += 1

    _write_manifest(dataset_root / "manifest.csv", samples)
    _write_source_manifest(dataset_root / "source_manifest.csv", selected)
    list_counts = _write_list_files(dataset_root, samples)
    audit_report = audit_dataset(dataset_root, write_report=True)
    yaml_paths = _dataset_yaml_paths(final_root, yaml_dir)

    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "selection_digest": selection_digest(selected),
        "source_image_digest": source_digest,
        "config": dict(config),
        "environment": environment_report(),
        "source_report": source_report,
        "materialization": dict(materialization),
        "list_counts": list_counts,
        "audit": audit_report,
        "yaml_paths": {name: str(path.resolve()) for name, path in yaml_paths.items()},
    }
    (dataset_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    write_dataset_yamls(final_root, yaml_dir)
    backup_root: Path | None = None
    if final_root.exists():
        backup_root = final_root.with_name(
            f".{final_root.name}.backup-{os.getpid()}-{time.time_ns()}"
        )
        final_root.rename(backup_root)
    try:
        dataset_root.rename(final_root)
    except OSError as promotion_error:
        if backup_root is not None:
            backup_root.rename(final_root)
        raise promotion_error
    staging.cleanup()
    if backup_root is not None:
        shutil.rmtree(backup_root)
    return metadata


def _read_yolo_label(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    classes: list[int] = []
    boxes: list[list[float]] = []
    text = path.read_text().strip()
    if not text:
        return np.empty((0,), dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed label {path}:{line_number}: {line}")
        cls = int(parts[0])
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:])
        if cls not in CLASS_NAMES:
            raise ValueError(f"Invalid class in {path}:{line_number}: {cls}")
        values = (x_center, y_center, box_width, box_height)
        if not all(np.isfinite(values)):
            raise ValueError(f"Non-finite label in {path}:{line_number}")
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        ):
            raise ValueError(f"Out-of-range label in {path}:{line_number}: {line}")
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        tolerance = 1e-4 * max(width, height)
        if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
            raise ValueError(f"Box exceeds image in {path}:{line_number}: {line}")
        classes.append(cls)
        boxes.append([max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32)


def audit_dataset(dataset_root: Path, write_report: bool = True) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.csv"
    source_manifest_path = dataset_root / "source_manifest.csv"
    manifest = read_manifest(manifest_path)
    if not manifest:
        raise ValueError("The dataset manifest is empty.")
    artifact_hasher = hashlib.sha256()
    for path in (manifest_path, source_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
        artifact_hasher.update(path.name.encode("utf-8"))
        artifact_hasher.update(sha256_file(path).encode("ascii"))
    sample_ids = [row["sample_id"] for row in manifest]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("The dataset manifest contains duplicate sample IDs.")

    expected_images = {row["image_path"] for row in manifest}
    expected_labels = {row["label_path"] for row in manifest}
    actual_images = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    actual_labels = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "labels").rglob("*.txt")
        if path.is_file()
    }
    if actual_images != expected_images:
        raise ValueError(
            "Image tree differs from manifest: "
            f"missing={len(expected_images - actual_images)}, orphan={len(actual_images - expected_images)}"
        )
    if actual_labels != expected_labels:
        raise ValueError(
            "Label tree differs from manifest: "
            f"missing={len(expected_labels - actual_labels)}, orphan={len(actual_labels - expected_labels)}"
        )

    for split in SPLITS:
        for suffix, source in (("", None), ("_coco", "coco"), ("_stanford", "stanford")):
            list_path = dataset_root / f"{split}{suffix}.txt"
            artifact_hasher.update(list_path.name.encode("utf-8"))
            artifact_hasher.update(sha256_file(list_path).encode("ascii"))
            lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
            normalized = [line[2:] if line.startswith("./") else line for line in lines]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Duplicate entries in {list_path}")
            expected = {
                row["image_path"]
                for row in manifest
                if row["split"] == split and (source is None or row["source"] == source)
            }
            if set(normalized) != expected:
                raise ValueError(f"{list_path} differs from the dataset manifest.")

    stats = Counter()
    split_stats: dict[str, Counter] = {split: Counter() for split in SPLITS}
    groups_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    paths_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    sha_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    dhash_groups: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    full_source_items: list[PerceptualItem] = []

    for row in manifest:
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"Unknown split in manifest: {split}")
        image_path = dataset_root / row["image_path"]
        label_path = dataset_root / row["label_path"]
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing pair: {image_path}, {label_path}")
        if row["image_path"] in paths_by_split[split]:
            raise ValueError(f"Duplicate image in {split}: {row['image_path']}")
        paths_by_split[split].add(row["image_path"])
        groups_by_split[split].add(row["group_id"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable generated image: {image_path}")
        height, width = image.shape[:2]
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Manifest size mismatch: {image_path}")
        classes, boxes = _read_yolo_label(label_path, width, height)
        if len(classes) != int(row["box_count"]):
            raise ValueError(f"Manifest box-count mismatch: {label_path}")
        if row["variant"] == "partial" and split != "train":
            raise ValueError(f"Augmented sample outside training split: {image_path}")

        stats["images"] += 1
        stats["objects"] += len(classes)
        stats[f"source_{row['source']}"] += 1
        stats[f"variant_{row['variant']}"] += 1
        split_stats[split]["images"] += 1
        split_stats[split]["objects"] += len(classes)
        split_stats[split][f"source_{row['source']}"] += 1
        if len(classes) == 0:
            split_stats[split]["negative_images"] += 1
        for cls in classes:
            split_stats[split][f"class_{int(cls)}_objects"] += 1
        for x1, y1, x2, y2 in boxes:
            normalized_width = (x2 - x1) / width
            normalized_height = (y2 - y1) / height
            if normalized_width >= 0.9999 and normalized_height >= 0.9999:
                split_stats[split]["full_frame_boxes"] += 1

        content_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path)
        artifact_hasher.update(row["image_path"].encode("utf-8"))
        artifact_hasher.update(content_hash.encode("ascii"))
        artifact_hasher.update(row["label_path"].encode("utf-8"))
        artifact_hasher.update(label_hash.encode("ascii"))
        perceptual_hash = _difference_hash(image)
        sha_groups[content_hash][split].append(row["image_path"])
        dhash_groups[perceptual_hash][split].append(row["image_path"])
        if row["variant"] == "full":
            full_source_items.append(
                PerceptualItem(
                    split=split,
                    key=row["group_id"],
                    image_path=image_path,
                    content_hash=content_hash,
                    difference_hash=perceptual_hash,
                )
            )

    source_overlap: dict[str, list[str]] = {}
    path_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}_{right}"
            source_overlap[key] = sorted(groups_by_split[left] & groups_by_split[right])
            path_overlap[key] = sorted(paths_by_split[left] & paths_by_split[right])

    exact_cross_split = {
        digest: dict(by_split)
        for digest, by_split in sha_groups.items()
        if sum(bool(by_split.get(split)) for split in SPLITS) > 1
    }
    perceptual_cross_split = {
        str(digest): dict(by_split)
        for digest, by_split in dhash_groups.items()
        if sum(bool(by_split.get(split)) for split in SPLITS) > 1
    }
    near_duplicate_clusters, verified_near_pairs = perceptual_duplicate_clusters(
        full_source_items
    )
    near_duplicate_cross_split = [
        [full_source_items[index].key for index in cluster]
        for cluster in near_duplicate_clusters
        if len({full_source_items[index].split for index in cluster}) > 1
    ]
    if any(source_overlap.values()):
        raise ValueError(f"Source-level leakage detected: {source_overlap}")
    if any(path_overlap.values()):
        raise ValueError(f"Path-level leakage detected: {path_overlap}")
    if exact_cross_split:
        raise ValueError(f"Exact image duplicates cross splits: {list(exact_cross_split)[:5]}")
    if near_duplicate_cross_split:
        raise ValueError(
            "Verified near-duplicate sources cross splits: "
            f"{near_duplicate_cross_split[:5]}"
        )

    split_reports = {split: dict(values) for split, values in split_stats.items()}
    for split, values in split_reports.items():
        positive_boxes = values.get("class_0_objects", 0) + values.get("class_1_objects", 0)
        values["full_frame_box_ratio"] = (
            values.get("full_frame_boxes", 0) / positive_boxes if positive_boxes else 0.0
        )

    report = {
        "totals": dict(stats),
        "splits": split_reports,
        "source_overlap": source_overlap,
        "path_overlap": path_overlap,
        "exact_cross_split_groups": len(exact_cross_split),
        "identical_dhash_cross_split_groups": len(perceptual_cross_split),
        "identical_dhash_examples": dict(list(perceptual_cross_split.items())[:20]),
        "near_duplicate_cross_split_groups": len(near_duplicate_cross_split),
        "verified_near_duplicate_pairs": verified_near_pairs,
        "artifact_digest": artifact_hasher.hexdigest(),
    }
    if write_report:
        (dataset_root / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def sample_preview_rows(
    dataset_root: Path,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    rows = [row for row in read_manifest(dataset_root / "manifest.csv") if row["split"] == split]
    rng = random.Random(stable_int(seed, "preview", split))
    return rng.sample(rows, min(count, len(rows)))


def draw_labelled_image(dataset_root: Path, row: Mapping[str, str]) -> np.ndarray:
    image_path = dataset_root / row["image_path"]
    label_path = dataset_root / row["label_path"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    height, width = image.shape[:2]
    classes, boxes = _read_yolo_label(label_path, width, height)
    for cls, (x1, y1, x2, y2) in zip(classes, boxes):
        color = (30, 210, 30) if cls == 0 else (30, 140, 255)
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.putText(
            image,
            CLASS_NAMES[int(cls)],
            (round(x1), max(15, round(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def training_arguments(
    data_yaml: Path,
    project: Path,
    name: str,
    device: str,
    mosaic: float,
) -> dict[str, Any]:
    return {
        "data": str(data_yaml.resolve()),
        "epochs": 60,
        "imgsz": 320,
        "batch": 16,
        "patience": 15,
        "device": device,
        "workers": 0 if device in {"cpu", "mps"} else 4,
        "project": str(project.resolve()),
        "name": name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "auto",
        "seed": 42,
        "deterministic": True,
        "rect": False,
        "val": True,
        "cache": False,
        "mosaic": mosaic,
        "close_mosaic": 10 if mosaic else 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 3.0,
        "translate": 0.1,
        "scale": 0.35,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "plots": True,
        "verbose": True,
        "resume": False,
    }


def dataset_identity(metadata: Mapping[str, Any]) -> dict[str, str]:
    fingerprint = metadata.get("fingerprint")
    artifact_digest = metadata.get("audit", {}).get("artifact_digest")
    if not isinstance(fingerprint, str) or not isinstance(artifact_digest, str):
        raise ValueError("Dataset metadata does not contain verified identity digests.")
    return {
        "fingerprint": fingerprint,
        "artifact_digest": artifact_digest,
    }


def _verified_dataset_binding(
    arguments: Mapping[str, Any],
    claimed_identity: Mapping[str, str],
) -> dict[str, Any]:
    data_value = arguments.get("data")
    if not isinstance(data_value, (str, Path)):
        raise ValueError("Training arguments must reference a dataset YAML.")
    data_yaml = Path(data_value).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    yaml_payload = yaml.safe_load(data_yaml.read_text())
    if not isinstance(yaml_payload, dict):
        raise ValueError(f"Invalid dataset YAML: {data_yaml}")

    root_value = yaml_payload.get("path")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"Dataset YAML does not define a path: {data_yaml}")
    dataset_root = Path(root_value)
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()

    metadata_path = dataset_root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(f"Dataset metadata schema is not current: {metadata_path}")

    expected_yaml = metadata.get("yaml_paths", {}).get("combined")
    if not isinstance(expected_yaml, str) or Path(expected_yaml).resolve() != data_yaml:
        raise RuntimeError(f"Training must use the verified combined dataset YAML: {data_yaml}")
    try:
        names = {int(key): str(value) for key, value in yaml_payload.get("names", {}).items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid class mapping in {data_yaml}") from error
    expected_yaml_values = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
    }
    if (
        names != CLASS_NAMES
        or any(yaml_payload.get(key) != value for key, value in expected_yaml_values.items())
    ):
        raise RuntimeError(f"Dataset YAML does not describe the verified combined split: {data_yaml}")

    stored_identity = dataset_identity(metadata)
    if any(claimed_identity.get(key) != value for key, value in stored_identity.items()):
        raise RuntimeError(f"Claimed dataset identity does not match {metadata_path}")
    live_audit = audit_dataset(dataset_root, write_report=False)
    if live_audit["artifact_digest"] != stored_identity["artifact_digest"]:
        raise RuntimeError(f"Dataset artifacts differ from their verified metadata: {dataset_root}")
    return {
        **stored_identity,
        "schema_version": DATASET_SCHEMA_VERSION,
        "root": str(dataset_root),
        "yaml": {
            "path": str(data_yaml),
            "sha256": sha256_file(data_yaml),
        },
    }


def run_binding_path(arguments: Mapping[str, Any]) -> Path:
    if "project" not in arguments or "name" not in arguments:
        raise ValueError("Training arguments must include project and name.")
    project = Path(str(arguments["project"])).resolve()
    name = str(arguments["name"])
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Training run name must be a single path component: {name!r}")
    return project / ".run-bindings" / f"{name}.json"


def _validate_binding_path(
    binding_path: Path,
    arguments: Mapping[str, Any],
) -> Path:
    expected = run_binding_path(arguments).resolve()
    if binding_path.resolve() != expected:
        raise ValueError(f"Run binding must be stored at {expected}, not {binding_path}")
    return expected


def _run_binding_identity(
    model: str | Path,
    arguments: Mapping[str, Any],
    identity: Mapping[str, str],
) -> dict[str, Any]:
    model_path = Path(model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return {
        "schema_version": RUN_BINDING_SCHEMA_VERSION,
        "dataset": _verified_dataset_binding(arguments, identity),
        "training_arguments": dict(arguments),
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "initial_model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _normalized_checkpoint_argument(key: str, value: Any) -> Any:
    if key in {"data", "project"} and value is not None:
        return str(Path(str(value)).resolve())
    if key == "device" and value is not None:
        rendered = str(value).lower()
        if rendered == "cuda":
            return "0"
        if rendered.startswith("cuda:"):
            return rendered.split(":", 1)[1]
        return rendered
    return value


def _verified_checkpoint_metadata(
    checkpoint: Path,
    binding: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], datetime]:
    checkpoint_data, _ = torch_safe_load(checkpoint)
    if not isinstance(checkpoint_data, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {checkpoint}")
    train_arguments = checkpoint_data.get("train_args")
    if not isinstance(train_arguments, dict):
        raise RuntimeError(f"Checkpoint has no training arguments: {checkpoint}")
    mismatches = []
    for key in RESUME_ARGUMENT_KEYS:
        expected = _normalized_checkpoint_argument(key, arguments.get(key))
        actual = _normalized_checkpoint_argument(key, train_arguments.get(key))
        if actual != expected:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            f"Checkpoint training arguments do not match its run binding: {mismatches}"
        )

    checkpoint_version = checkpoint_data.get("version")
    if checkpoint_version != ultralytics.__version__:
        raise RuntimeError(
            f"Checkpoint Ultralytics version {checkpoint_version!r} does not match "
            f"{ultralytics.__version__!r}."
        )

    checkpoint_date_value = checkpoint_data.get("date")
    created_at_value = binding.get("created_at")
    try:
        checkpoint_date = datetime.fromisoformat(str(checkpoint_date_value).replace("Z", "+00:00"))
        created_at = datetime.fromisoformat(str(created_at_value).replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"Checkpoint or binding has an invalid timestamp: {checkpoint}") from error
    checkpoint_date = checkpoint_date.astimezone(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)
    if checkpoint_date < created_at:
        raise RuntimeError(f"Checkpoint predates its run binding: {checkpoint}")
    return checkpoint_data, checkpoint_date


def _pending_checkpoint_record(
    checkpoint: Path,
    binding: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_data, checkpoint_date = _verified_checkpoint_metadata(
        checkpoint,
        binding,
        arguments,
    )
    epoch = checkpoint_data.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError(f"Checkpoint is not an interrupted training state: {checkpoint}")

    previous = binding.get("resume_checkpoint")
    if isinstance(previous, dict):
        previous_epoch = previous.get("epoch")
        if isinstance(previous_epoch, int) and epoch < previous_epoch:
            raise RuntimeError(f"Checkpoint epoch regressed from {previous_epoch} to {epoch}.")
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "epoch": epoch,
        "date": checkpoint_date.isoformat(),
    }


def _completed_checkpoint_record(
    checkpoint: Path,
    binding: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_data, checkpoint_date = _verified_checkpoint_metadata(
        checkpoint,
        binding,
        arguments,
    )
    epoch = checkpoint_data.get("epoch")
    if not isinstance(epoch, int):
        raise RuntimeError(f"Checkpoint has an invalid epoch: {checkpoint}")
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "epoch": epoch,
        "date": checkpoint_date.isoformat(),
    }


def prepare_run_binding(
    binding_path: Path,
    model: str | Path,
    arguments: Mapping[str, Any],
    identity: Mapping[str, str],
) -> dict[str, Any]:
    binding_path = _validate_binding_path(binding_path, arguments)
    expected = _run_binding_identity(model, arguments, identity)
    if binding_path.exists():
        existing = json.loads(binding_path.read_text())
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Run binding conflicts with this experiment: {binding_path}")
        if existing.get("status") != "pending":
            raise RuntimeError(f"Run binding is already finalized: {binding_path}")
        return existing

    payload = {
        **expected,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(binding_path, payload)
    return payload


def finalize_run_binding(binding_path: Path) -> dict[str, Any]:
    if not binding_path.exists():
        raise FileNotFoundError(binding_path)
    binding = json.loads(binding_path.read_text())
    arguments = binding.get("training_arguments", {})
    binding_path = _validate_binding_path(binding_path, arguments)
    if binding.get("status") != "pending":
        raise RuntimeError(f"Run binding is not pending: {binding_path}")
    live_dataset = _verified_dataset_binding(arguments, binding.get("dataset", {}))
    if live_dataset != binding.get("dataset"):
        raise RuntimeError(f"Dataset changed while training the bound run: {binding_path}")
    expected_runtime = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }
    if binding.get("runtime") != expected_runtime:
        raise RuntimeError(f"Runtime changed while training the bound run: {binding_path}")
    run_directory = Path(str(arguments["project"])) / str(arguments["name"])
    checkpoints = {}
    for name in ("best.pt", "last.pt"):
        checkpoint = run_directory / "weights" / name
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoints[name] = _completed_checkpoint_record(
            checkpoint,
            binding,
            arguments,
        )
    binding.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoints": checkpoints,
        }
    )
    _write_json_atomic(binding_path, binding)
    return binding


def verify_checkpoint_binding(
    checkpoint: Path,
    binding_path: Path,
    model: str | Path,
    arguments: Mapping[str, Any],
    identity: Mapping[str, str],
    allow_pending_last: bool = False,
) -> dict[str, Any]:
    if not binding_path.exists():
        raise FileNotFoundError(f"Missing run binding: {binding_path}")
    binding_path = _validate_binding_path(binding_path, arguments)
    binding = json.loads(binding_path.read_text())
    expected = _run_binding_identity(model, arguments, identity)
    if any(binding.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Checkpoint run does not match the current dataset: {binding_path}")

    run_directory = Path(str(arguments["project"])) / str(arguments["name"])
    resolved_checkpoint = checkpoint.resolve()
    expected_checkpoint = (run_directory / "weights" / checkpoint.name).resolve()
    if checkpoint.name not in {"best.pt", "last.pt"} or resolved_checkpoint != expected_checkpoint:
        raise ValueError(f"Checkpoint is outside the bound run: {checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    status = binding.get("status")
    if status == "pending":
        if not allow_pending_last or checkpoint.name != "last.pt":
            raise RuntimeError(f"Run binding is not finalized: {binding_path}")
        binding["resume_checkpoint"] = _pending_checkpoint_record(
            checkpoint,
            binding,
            arguments,
        )
        _write_json_atomic(binding_path, binding)
    elif status == "complete":
        if allow_pending_last:
            raise RuntimeError(f"Resume requires a pending run binding: {binding_path}")
        checkpoint_record = binding.get("checkpoints", {}).get(checkpoint.name)
        if (
            not isinstance(checkpoint_record, dict)
            or checkpoint_record.get("path") != str(resolved_checkpoint)
            or checkpoint_record.get("sha256") != sha256_file(checkpoint)
        ):
            raise RuntimeError(f"Checkpoint digest does not match its run binding: {checkpoint}")
    else:
        raise RuntimeError(f"Invalid run-binding status in {binding_path}")
    return binding


def format_yolo_cli(
    model: str | Path,
    arguments: Mapping[str, Any],
    dataset_root: Path,
    identity: Mapping[str, str],
) -> str:
    model_value = str(model)
    run_directory = Path(str(arguments["project"])) / str(arguments["name"])
    binding_path = run_binding_path(arguments)
    quoted_run = shlex.quote(str(run_directory))
    message = shlex.quote(f"Run already exists: {run_directory}")
    prepare_code = (
        "from pathlib import Path; "
        "from quadruped_pipeline import prepare_run_binding; "
        f"prepare_run_binding(Path({str(binding_path)!r}), {model_value!r}, "
        f"{dict(arguments)!r}, {dict(identity)!r})"
    )
    train_code = (
        "from ultralytics import YOLO; "
        f"YOLO({model_value!r}).train(**{dict(arguments)!r})"
    )
    finalize_code = (
        "from pathlib import Path; "
        "from quadruped_pipeline import finalize_run_binding; "
        f"finalize_run_binding(Path({str(binding_path)!r}))"
    )
    python = shlex.quote(sys.executable)
    labels_directory = shlex.quote(str((dataset_root / "labels").resolve()))
    commands = [
        f"(test ! -e {quoted_run} || (echo {message} >&2 && false))",
        f"find {labels_directory} -type f -name '*.cache' -delete",
        f"{python} -c {shlex.quote(prepare_code)}",
        f"{python} -c {shlex.quote(train_code)}",
    ]
    commands.append(f"{python} -c {shlex.quote(finalize_code)}")
    return " && \\\n".join(commands)


def summarize_metrics(metrics: DetMetrics) -> dict[str, Any]:
    summary = {
        "aggregate": metrics.results_dict,
        "per_class": metrics.summary(),
    }
    if any(float(value) > 0.0 for value in metrics.speed.values()):
        summary["speed_ms"] = metrics.speed
    return summary


def summarize_ap_metrics(metrics: DetMetrics) -> dict[str, Any]:
    return {
        "aggregate": {
            name: value for name, value in metrics.results_dict.items() if "mAP" in name
        },
        "per_class": [
            {
                name: value
                for name, value in row.items()
                if name in {"Class", "Images", "Instances"} or "mAP" in name
            }
            for row in metrics.summary()
        ],
    }


def validate_without_metric_logging(model: YOLO, **kwargs: Any) -> DetMetrics:
    previous_level = LOGGER.level
    LOGGER.setLevel(logging.WARNING)
    try:
        kwargs.setdefault("verbose", False)
        return model.val(**kwargs)
    finally:
        LOGGER.setLevel(previous_level)


def best_f1_confidence(metrics: DetMetrics) -> float:
    if not hasattr(metrics.box, "f1_curve") or not len(metrics.box.f1_curve):
        raise ValueError("Validation metrics do not contain an F1-confidence curve.")
    mean_curve = np.asarray(metrics.box.f1_curve).mean(axis=0)
    return float(np.asarray(metrics.box.px)[int(mean_curve.argmax())])


def resolve_image_list(dataset_root: Path, list_path: Path) -> list[Path]:
    images = []
    for line in list_path.read_text().splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("./"):
            images.append(dataset_root / value[2:])
        else:
            path = Path(value)
            images.append(path if path.is_absolute() else dataset_root / path)
    return images


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(dataset_root)
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ValueError(f"Image is outside the expected images tree: {image_path}")
    parts[0] = "labels"
    return (dataset_root / Path(*parts)).with_suffix(".txt")


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    thresholds: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros((pred_classes.shape[0], thresholds.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes[None, :]
    class_iou = (iou * correct_class).cpu().numpy()
    for index, threshold in enumerate(thresholds.cpu().tolist()):
        matches = np.array(np.nonzero(class_iou >= threshold)).T
        if not len(matches):
            continue
        if len(matches) > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), index] = True
    return correct


def _validation_letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _square_letterbox(image, boxes, image_size, scaleup=False)


def evaluate_coco_pretrained_baseline(
    model_path: str | Path,
    dataset_root: Path,
    list_path: Path,
    device: str,
    image_size: int = 320,
    batch: int = 16,
    plot_dir: Path | None = None,
) -> DetMetrics:
    if plot_dir is not None and plot_dir.exists():
        raise FileExistsError(f"Baseline output already exists: {plot_dir}")
    if batch < 1:
        raise ValueError("batch must be at least 1.")
    images = resolve_image_list(dataset_root, list_path)
    model = YOLO(str(model_path))
    network, inference_device = _pytorch_detection_network(model, device)
    metrics = DetMetrics(names=CLASS_NAMES)
    thresholds = torch.linspace(0.5, 0.95, 10)

    for start in range(0, len(images), batch):
        image_chunk = images[start : start + batch]
        frames = []
        targets = []
        for image_path in image_chunk:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable evaluation image: {image_path}")
            height, width = image.shape[:2]
            target_classes, target_boxes = _read_yolo_label(
                label_path_for_image(dataset_root, image_path),
                width,
                height,
            )
            frame, transformed_boxes = _validation_letterbox(
                image,
                target_boxes,
                image_size,
            )
            frames.append(frame)
            targets.append((image_path, target_classes, transformed_boxes))

        predictions = _multi_label_predictions(
            network,
            frames,
            inference_device,
            class_count=len(model.names),
            confidence=0.001,
            iou=0.7,
            max_detections=300,
            classes=[0, 16],
        )
        for (image_path, target_classes, target_boxes), prediction in zip(
            targets,
            predictions,
            strict=True,
        ):
            if not len(prediction):
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_classes = np.empty((0,), dtype=np.int64)
                confidences = np.empty((0,), dtype=np.float32)
            else:
                pred_boxes = prediction[:, :4].cpu().numpy().astype(np.float32)
                original_classes = prediction[:, 5].cpu().numpy().astype(np.int64)
                pred_classes = np.where(original_classes == 0, 0, 1).astype(np.int64)
                confidences = prediction[:, 4].cpu().numpy().astype(np.float32)

            if len(target_boxes) and len(pred_boxes):
                iou = box_iou(torch.from_numpy(target_boxes), torch.from_numpy(pred_boxes))
                correct = _match_predictions(
                    torch.from_numpy(pred_classes),
                    torch.from_numpy(target_classes),
                    iou,
                    thresholds,
                )
            else:
                correct = np.zeros((len(pred_boxes), len(thresholds)), dtype=bool)

            metrics.update_stats(
                {
                    "tp": correct,
                    "conf": confidences,
                    "pred_cls": pred_classes,
                    "target_cls": target_classes,
                    "target_img": np.unique(target_classes),
                    "im_name": image_path.name,
                }
            )

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    metrics.process(save_dir=plot_dir or Path("."), plot=plot_dir is not None)
    return metrics


def negative_false_positive_rate(
    model: YOLO,
    dataset_root: Path,
    split: str,
    confidence: float,
    device: str,
    image_size: int = 320,
    batch: int = 16,
) -> dict[str, float | int]:
    if batch < 1:
        raise ValueError("batch must be at least 1.")
    rows = [
        row
        for row in read_manifest(dataset_root / "manifest.csv")
        if row["split"] == split and int(row["box_count"]) == 0
    ]
    images = [dataset_root / row["image_path"] for row in rows]
    if not images:
        return {
            "negative_images": 0,
            "images_with_detection": 0,
            "detections": 0,
            "image_false_positive_rate": 0.0,
            "detections_per_image": 0.0,
            "confidence": confidence,
        }
    network, inference_device = _pytorch_detection_network(model, device)
    images_with_detection = 0
    detection_count = 0
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    for start in range(0, len(images), batch):
        frames = []
        for image_path in images[start : start + batch]:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Unreadable evaluation image: {image_path}")
            frame, _ = _validation_letterbox(
                image,
                empty_boxes,
                image_size,
            )
            frames.append(frame)
        predictions = _multi_label_predictions(
            network,
            frames,
            inference_device,
            class_count=len(model.names),
            confidence=confidence,
            iou=0.7,
            max_detections=300,
        )
        for prediction in predictions:
            count = len(prediction)
            detection_count += count
            images_with_detection += int(count > 0)
    total = len(images)
    return {
        "negative_images": total,
        "images_with_detection": images_with_detection,
        "detections": detection_count,
        "image_false_positive_rate": images_with_detection / total if total else 0.0,
        "detections_per_image": detection_count / total if total else 0.0,
        "confidence": confidence,
    }


def benchmark_batch_one(
    model: YOLO,
    image_path: Path,
    device: str,
    image_size: int = 320,
    warmup: int = 5,
    iterations: int = 30,
) -> dict[str, float | str]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unreadable benchmark image: {image_path}")
    height, width = image.shape[:2]
    side = min(height, width)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    square = image[y1 : y1 + side, x1 : x1 + side]
    benchmark_frame = cv2.resize(
        square,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA if side > image_size else cv2.INTER_LINEAR,
    )

    def synchronize() -> None:
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        model.predict(
            source=benchmark_frame,
            imgsz=image_size,
            device=device,
            batch=1,
            rect=False,
            verbose=False,
        )
    synchronize()

    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        model.predict(
            source=benchmark_frame,
            imgsz=image_size,
            device=device,
            batch=1,
            rect=False,
            verbose=False,
        )
        synchronize()
        durations.append(time.perf_counter() - started)
    milliseconds = np.asarray(durations) * 1000.0
    return {
        "platform": platform.platform(),
        "device": device,
        "image_size": str(image_size),
        "input": "in-memory square frame",
        "iterations": float(iterations),
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "fps_from_mean": float(1000.0 / milliseconds.mean()),
    }
