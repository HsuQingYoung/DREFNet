"""Optional frame exclusions for training and evaluation.

Load an externally supplied JSON file of frames to skip, keyed by
``(dataset, split, sequence, frame)``. No quality-scoring model is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

FrameKey = Tuple[str, str, str]  # (split, sequence, frame)

SPLITS = ("train", "val", "test")


def dataset_name_for_root(root) -> str:
    """Return the ``exclude.json`` dataset key for a mask root.

    For example, ``.../dsec-emerge-3/dsec`` resolves to ``dsec-emerge-3``.
    """
    path = Path(root).expanduser()
    return path.parent.name if path.name == "dsec" else path.name


def exclusion_mask_for_samples(
    exclusions: Optional["RgbExclusions"],
    dataset_names: Sequence[Optional[str]],
    split: str,
    sample_sequences: Sequence[str],
    sample_image_indices: Sequence[int],
    frame_names: Mapping[str, Sequence[Sequence[str]]],
) -> Optional[np.ndarray]:
    """``(n_roots, n_samples)`` mask of samples whose alternate frame was flagged.

    One row per alternate RGB root. ``dataset_names[root_index]`` is the ``exclude.json``
    key of that root (``.../dsec-emerge-3/dsec`` -> ``dsec-emerge-3``);
    ``frame_names[sequence][root_index][image_index]`` is the frame file name of the
    alternate RGB source, i.e. exactly the name the flagged frame is recorded under.

    Returns ``None`` when nothing can be excluded (no exclusions / no roots), so callers
    can keep the previous behaviour with a single ``is None`` check.
    """
    if exclusions is None or not exclusions or not dataset_names:
        return None
    known = set(exclusions.datasets)
    mask = np.zeros((len(dataset_names), len(sample_sequences)), dtype=bool)
    for root_index, dataset_name in enumerate(dataset_names):
        if dataset_name not in known:
            continue
        for sample_index in range(len(sample_sequences)):
            sequence = sample_sequences[sample_index]
            per_root = frame_names.get(sequence)
            if not per_root or root_index >= len(per_root):
                continue
            frame_name = per_root[root_index][int(sample_image_indices[sample_index])]
            if exclusions.contains(dataset_name, split, sequence, frame_name):
                mask[root_index, sample_index] = True
    return mask


def reroute_flagged_samples(
    use_alt_rgb_for_sample: np.ndarray,
    alt_rgb_root_index_for_sample: np.ndarray,
    mask: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the inpainted RGB, but never from a root that flagged this frame.

    A frame is the *same* frame in all three DSEC-Emerge sets (same name, same events, same
    labels), so when the sampled root flagged it we can simply take the same frame from one
    of the roots that did not - the RGB still matches the events and labels. Roots whose
    dataset is unknown to ``exclude.json`` count as clean.

    Samples that were flagged by *every* root have no clean inpainted version at all; they
    fall back to the original RGB (``use_alt_rgb_for_sample = False``). The realized
    original:emergence ratio therefore ends up slightly below the requested
    ``alt_rgb_sampling_ratio`` - the caller reports the actual number.
    """
    if mask is None:
        return use_alt_rgb_for_sample, alt_rgb_root_index_for_sample
    use_alt = use_alt_rgb_for_sample.copy()
    root_index = alt_rgb_root_index_for_sample.copy()
    sample_indices = np.arange(len(use_alt))
    needs_reroute = np.nonzero(use_alt & mask[root_index, sample_indices])[0]
    for sample_index in needs_reroute:
        clean_roots = np.nonzero(~mask[:, sample_index])[0]
        if clean_roots.size == 0:
            use_alt[sample_index] = False
        else:
            root_index[sample_index] = int(rng.choice(clean_roots))
    return use_alt, root_index


class RgbExclusions:
    """Lookup table of flagged frames: ``(split, sequence, frame) -> {dataset: info}``.

    A frame can be flagged for one dataset and clean for another, because the three
    DSEC-Emerge sets remove different targets from the same frame. ``contains`` asks
    about one dataset; ``contains_any`` ignores the dataset (the union used when several
    RGB variants must be scored on identical frames).
    """

    def __init__(
        self,
        frames: Optional[Mapping[FrameKey, Mapping[str, Mapping]]] = None,
        meta: Optional[Mapping] = None,
    ):
        self._frames: Dict[FrameKey, Dict[str, Mapping]] = {
            (str(split), str(sequence), str(frame)): {str(name): dict(info or {}) for name, info in per_dataset.items()}
            for (split, sequence, frame), per_dataset in (frames or {}).items()
        }
        self.meta: Dict[str, object] = dict(meta or {})
        self._cache: Dict[Tuple[Optional[str], str, str], frozenset] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_payload(cls, payload) -> "RgbExclusions":
        """Build from a JSON object containing ``datasets`` or a bare dataset map."""
        if not isinstance(payload, Mapping):
            raise ValueError("exclude.json must contain a JSON object")
        datasets = payload.get("datasets", payload)
        if not isinstance(datasets, Mapping):
            raise ValueError("exclude.json 'datasets' must be a JSON object")

        frames: Dict[FrameKey, Dict[str, Mapping]] = {}
        for dataset_name, splits in datasets.items():
            for split, sequences in (splits or {}).items():
                for sequence, frame_map in (sequences or {}).items():
                    for frame, info in (frame_map or {}).items():
                        key = (str(split), str(sequence), str(frame))
                        frames.setdefault(key, {})[str(dataset_name)] = dict(info or {})
        meta = payload.get("meta") if isinstance(payload, Mapping) else None
        return cls(frames, meta)

    @classmethod
    def from_json(cls, path) -> "RgbExclusions":
        return cls.from_payload(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))

    # ---------------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        return bool(self._frames)

    @property
    def datasets(self) -> list:
        return sorted({name for per_dataset in self._frames.values() for name in per_dataset})

    def contains(self, dataset: str, split: str, sequence: str, frame: str) -> bool:
        """True when ``frame`` was flagged for ``dataset``."""
        per_dataset = self._frames.get((str(split), str(sequence), str(frame)))
        return bool(per_dataset) and dataset in per_dataset

    def contains_any(self, split: str, sequence: str, frame: str) -> bool:
        """True when ``frame`` was flagged for at least one dataset."""
        return bool(self._frames.get((str(split), str(sequence), str(frame))))

    def info(self, dataset: str, split: str, sequence: str, frame: str) -> Optional[Mapping]:
        per_dataset = self._frames.get((str(split), str(sequence), str(frame)))
        return None if not per_dataset else per_dataset.get(dataset)

    def frame_names(self, split: str, sequence: str, dataset: Optional[str] = None) -> frozenset:
        """All flagged frame names of one sequence, for one dataset or for the union."""
        cache_key = (dataset, str(split), str(sequence))
        cached = self._cache.get(cache_key)
        if cached is None:
            names = set()
            for (key_split, key_sequence, frame), per_dataset in self._frames.items():
                if key_split != str(split) or key_sequence != str(sequence):
                    continue
                if dataset is None or dataset in per_dataset:
                    names.add(frame)
            cached = frozenset(names)
            self._cache[cache_key] = cached
        return cached

    def sequences(self, dataset: Optional[str] = None, split: Optional[str] = None) -> list:
        return sorted(
            {
                (key_split, key_sequence)
                for (key_split, key_sequence, _), per_dataset in self._frames.items()
                if (split is None or key_split == split) and (dataset is None or dataset in per_dataset)
            }
        )

    def counts(self) -> Dict[str, Dict[str, int]]:
        """``dataset -> split -> number of flagged frames`` (plus an ``"ALL"`` row)."""
        counts: Dict[str, Dict[str, int]] = {}
        for (split, _sequence, _frame), per_dataset in self._frames.items():
            for dataset_name in per_dataset:
                counts.setdefault(dataset_name, {}).setdefault(split, 0)
                counts[dataset_name][split] += 1
        counts["ALL"] = {split: sum(row.get(split, 0) for name, row in counts.items() if name != "ALL") for split in SPLITS}
        return counts
