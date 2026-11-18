from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Callable, Dict, List

from torch_geometric.data import Dataset

import numpy as np
import cv2

import torch
from functools import lru_cache

from dsec_det.dataset import DSECDet

from dsec_det.io import yaml_file_to_dict
from .dsec_utils import filter_tracks, crop_tracks, rescale_tracks, compute_class_mapping, map_classes, filter_small_bboxes
from dsec_det.directory import BaseDirectory
from .augment import init_transforms
from .rgb_exclusions import RgbExclusions, dataset_name_for_root, exclusion_mask_for_samples, reroute_flagged_samples
from .utils import to_data

from ..visualization.bbox_viz import draw_bbox_on_img
from ..visualization.event_viz import draw_events_on_image


def tracks_to_array(tracks):
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)


def filter_tracks_by_track_ids(tracks, track_ids):
    if track_ids is None:
        return tracks
    if len(track_ids) == 0 or len(tracks) == 0:
        return tracks[:0].copy()
    track_ids = np.asarray(sorted(track_ids), dtype=tracks["track_id"].dtype)
    return tracks[np.isin(tracks["track_id"], track_ids)]



def interpolate_tracks(detections_0, detections_1, t):
    assert len(detections_1) == len(detections_0)
    if len(detections_0) == 0:
        return detections_1

    t0 = detections_0['t'][0]
    t1 = detections_1['t'][0]

    assert t0 < t1

    # need to sort detections
    detections_0 = detections_0[detections_0['track_id'].argsort()]
    detections_1 = detections_1[detections_1['track_id'].argsort()]

    r = ( t - t0 ) / ( t1 - t0 )
    detections_out = detections_0.copy()
    for k in 'xywh':
        detections_out[k] = detections_0[k] * (1 - r) + detections_1[k] * r

    return detections_out


def interpolate_shared_tracks(detections_0, detections_1, t):
    if len(detections_0) == 0 or len(detections_1) == 0:
        return detections_0[:0].copy()

    detections_0 = detections_0[detections_0["track_id"].argsort()]
    detections_1 = detections_1[detections_1["track_id"].argsort()]
    _, idx0, idx1 = np.intersect1d(
        detections_0["track_id"],
        detections_1["track_id"],
        assume_unique=False,
        return_indices=True,
    )

    if len(idx0) == 0:
        return detections_0[:0].copy()

    detections_0 = detections_0[idx0]
    detections_1 = detections_1[idx1]
    return interpolate_tracks(detections_0, detections_1, t)

class EventDirectory(BaseDirectory):
    @property
    @lru_cache
    def event_file(self):
        return self.root / "left/events_2x.h5"


class DSEC(Dataset):
    MAPPING = dict(pedestrian="pedestrian", rider=None, car="car", bus="car", truck="car", bicycle=None,
                   motorcycle=None, train=None)
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable]=None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 num_classes: int=2,
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False,
                 alt_rgb_root: Optional[Path] = None,
                 alt_rgb_roots: Optional[list[Path]] = None,
                 alt_rgb_sampling_ratio: float = 0.0,
                 rgb_sampling_weights: Optional[list[float]] = None,
                 rgb_sampling_seed: int = 0,
                 interframe_offsets_us: Optional[list[int]] = None,
                 interframe_ratios: Optional[list[float]] = None,
                 randomize_interframe_offsets: bool = False,
                 interframe_offset_seed: int = 0,
                 shared_track_interpolation: bool = False,
                 removed_targets_only_for_alt: bool = False,
                 alt_rgb_exclude_json: Optional[Path] = None,
                 emergence_mask_enabled: bool = False):

        Dataset.__init__(self)

        self.split = split

        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        for directory in self.dataset.directories.values():
            directory.events = EventDirectory(directory.events.root)

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        if num_classes == 2:
            self.classes = ("car", "pedestrian")
            class_mapping = self.MAPPING
        elif num_classes == 8:
            self.classes = tuple(self.dataset.classes)
            if len(self.classes) != 8:
                raise ValueError(f"Expected 8 native DSEC classes, got {len(self.classes)}: {self.classes}")
            class_mapping = {class_name: class_name for class_name in self.classes}
        else:
            raise ValueError(f"DSEC supports 2 or 8 training classes, got {num_classes}")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.debug = debug
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, class_mapping)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval
        self.alt_rgb_sampling_ratio = float(alt_rgb_sampling_ratio)
        self.primary_rgb_sampling_ratio = 1.0 - self.alt_rgb_sampling_ratio
        self.rgb_sampling_seed = int(rgb_sampling_seed)
        self.alt_rgb_roots = self._normalize_alt_rgb_roots(alt_rgb_roots, alt_rgb_root)
        self.rgb_sampling_weights = self._normalize_rgb_sampling_weights(rgb_sampling_weights)
        self.primary_rgb_image_files = self._build_primary_rgb_image_files()
        self.alt_rgb_image_files = self._build_alt_rgb_image_files()
        self.interframe_offsets_us = self._normalize_interframe_offsets(interframe_offsets_us)
        self.interframe_ratios = self._normalize_interframe_ratios(interframe_ratios)
        if self.interframe_offsets_us is not None and self.interframe_ratios is not None:
            raise ValueError("Specify either interframe_offsets_us or interframe_ratios, not both")
        self.randomize_interframe_offsets = bool(randomize_interframe_offsets)
        self.interframe_offset_seed = int(interframe_offset_seed)
        self.shared_track_interpolation = bool(shared_track_interpolation)
        self.removed_targets_only_for_alt = bool(removed_targets_only_for_alt)
        self.emergence_mask_enabled = bool(emergence_mask_enabled)
        self.alt_removed_track_ids = self._build_alt_removed_track_ids()
        self.alt_rgb_exclude_json = Path(alt_rgb_exclude_json).expanduser() if alt_rgb_exclude_json else None
        self.alt_rgb_exclusions = (
            RgbExclusions.from_json(self.alt_rgb_exclude_json) if self.alt_rgb_exclude_json else None
        )

        if self.no_eval:
            only_perfect_tracks = False

        self.image_index_pairs, self.track_masks = filter_tracks(dataset=self.dataset, image_width=self.width,
                                                                 image_height=self.height,
                                                                 class_remapping=self.class_remapping,
                                                                 min_bbox_height=min_bbox_height,
                                                                 min_bbox_diag=min_bbox_diag,
                                                                 only_perfect_tracks=only_perfect_tracks,
                                                                 scale=scale)
        self.sample_sequence_names, self.sample_image_indices = self._build_sample_index()
        self.alt_rgb_excluded_per_root = self._build_alt_rgb_excluded_per_root()
        self.use_alt_rgb_for_sample = np.zeros(len(self.sample_sequence_names), dtype=bool)
        self.alt_rgb_root_index_for_sample = np.zeros(len(self.sample_sequence_names), dtype=np.int64)
        self.sample_interframe_offsets_us = None
        self.sample_interframe_ratios = None
        self.resample_image_sources(epoch=0)
        self.resample_interframe_offsets(epoch=0)

    def _build_primary_rgb_image_files(self):
        rgb_files = {}
        for name, directory in self.dataset.directories.items():
            rgb_files[name] = directory.images.image_files_distorted
        return rgb_files

    def _normalize_alt_rgb_roots(self, alt_rgb_roots: Optional[list[Path]], alt_rgb_root: Optional[Path]) -> list[Path]:
        raw_roots = []
        if alt_rgb_roots is not None:
            raw_roots.extend(Path(root).expanduser() for root in alt_rgb_roots)
        elif alt_rgb_root is not None:
            raw_roots.append(Path(alt_rgb_root).expanduser())

        normalized_roots = []
        for root in raw_roots:
            if root.name != "dsec":
                root = root / "dsec"
            normalized_roots.append(root)
        return normalized_roots

    def _build_alt_rgb_image_files(self):
        if not self.alt_rgb_roots or (self.alt_rgb_sampling_ratio <= 0.0 and self.rgb_sampling_weights is None):
            return {}

        rgb_files = {}
        for name, directory in self.dataset.directories.items():
            default_files = directory.images.image_files_distorted
            alt_files_per_root = []
            for alt_rgb_root in self.alt_rgb_roots:
                alt_split_root = alt_rgb_root / self.dataset.root.name
                alt_files = sorted(list((alt_split_root / directory.root.name / "images/left/distorted").glob("*.png")))
                if len(alt_files) != len(default_files):
                    raise ValueError(
                        f"Alternate RGB file count mismatch for sequence {name} in {alt_rgb_root}: "
                        f"{len(alt_files)} vs {len(default_files)}"
                    )
                alt_files_per_root.append(alt_files)
            rgb_files[name] = alt_files_per_root
        return rgb_files

    def _normalize_rgb_sampling_weights(self, weights: Optional[list[float]]) -> Optional[np.ndarray]:
        if weights is None:
            return None
        if not self.alt_rgb_roots:
            raise ValueError("rgb_sampling_weights requires at least one alternate RGB root")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) != len(self.alt_rgb_roots) + 1:
            raise ValueError(
                "rgb_sampling_weights must contain one weight for original RGB followed by one for each alternate root"
            )
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("rgb_sampling_weights must be non-negative and have a positive sum")
        return weights / weights.sum()

    def _build_alt_removed_track_ids(self):
        if not self.alt_rgb_roots:
            return []

        removed_track_ids_per_root = []
        for alt_rgb_root in self.alt_rgb_roots:
            split_root = alt_rgb_root / self.dataset.root.name
            per_sequence = {}
            for name in self.dataset.directories.keys():
                json_path = split_root / name / "images" / "left" / "distorted_removed_targets.json"
                frame_to_ids = {}
                if json_path.exists():
                    with json_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    for frame_info in payload.get("frames", []):
                        frame_name = str(frame_info.get("frame_name", "")).strip()
                        removed_targets = frame_info.get("removed_targets", []) or []
                        frame_to_ids[frame_name] = {
                            int(item["track_id"])
                            for item in removed_targets
                            if "track_id" in item
                        }
                per_sequence[name] = frame_to_ids
            removed_track_ids_per_root.append(per_sequence)
        return removed_track_ids_per_root

    def _normalize_interframe_offsets(self, offsets: Optional[list[int]]) -> Optional[np.ndarray]:
        if offsets is None:
            return None
        offsets = np.asarray(offsets, dtype=np.int64)
        offsets = offsets[offsets > 0]
        offsets = np.unique(offsets)
        if offsets.size == 0:
            return None
        return offsets

    def _normalize_interframe_ratios(self, ratios: Optional[list[float]]) -> Optional[np.ndarray]:
        if ratios is None:
            return None
        ratios = np.asarray(ratios, dtype=np.float64)
        if ratios.ndim != 1 or np.any(ratios <= 0.0) or np.any(ratios > 1.0):
            raise ValueError("interframe_ratios must be within (0, 1]")
        ratios = np.unique(ratios)
        return ratios if ratios.size > 0 else None

    def _build_sample_index(self):
        sample_sequence_names = []
        sample_image_indices = []
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            for image_index_0, _ in self.image_index_pairs[name]:
                sample_sequence_names.append(name)
                sample_image_indices.append(int(image_index_0))
        return sample_sequence_names, np.asarray(sample_image_indices, dtype=np.int64)

    def resample_image_sources(self, epoch: int):
        rng = np.random.default_rng(self.rgb_sampling_seed + int(epoch))
        if not self.alt_rgb_image_files:
            self.use_alt_rgb_for_sample = np.zeros(len(self.sample_sequence_names), dtype=bool)
            self.alt_rgb_root_index_for_sample = np.zeros(len(self.sample_sequence_names), dtype=np.int64)
        elif self.rgb_sampling_weights is not None:
            choices = rng.choice(len(self.alt_rgb_roots) + 1, size=len(self.sample_sequence_names), p=self.rgb_sampling_weights)
            self.use_alt_rgb_for_sample = choices > 0
            self.alt_rgb_root_index_for_sample = np.maximum(choices - 1, 0).astype(np.int64)
        elif self.alt_rgb_sampling_ratio <= 0.0:
            self.use_alt_rgb_for_sample = np.zeros(len(self.sample_sequence_names), dtype=bool)
            self.alt_rgb_root_index_for_sample = np.zeros(len(self.sample_sequence_names), dtype=np.int64)
        else:
            self.use_alt_rgb_for_sample = rng.random(len(self.sample_sequence_names)) >= self.primary_rgb_sampling_ratio
            self.alt_rgb_root_index_for_sample = rng.integers(low=0, high=len(self.alt_rgb_roots), size=len(self.sample_sequence_names))
        self._avoid_flagged_alt_rgb(rng)

    def _build_alt_rgb_excluded_per_root(self):
        """``(n_alt_roots, n_samples)`` mask of samples whose alternate frame was flagged.

        The RC-S filter is keyed by ``(dataset, split, sequence, frame)``; the dataset name
        comes from the alternate root directory (for example,
        ``.../dsec-emerge-3/dsec`` -> ``dsec-emerge-3``) and the split from the
        config split of this dataset.
        """
        if self.alt_rgb_exclusions is None or not self.alt_rgb_roots:
            return None
        frame_names = {
            sequence: [[path.name for path in files] for files in per_root]
            for sequence, per_root in self.alt_rgb_image_files.items()
        }
        dataset_names = [dataset_name_for_root(root) for root in self.alt_rgb_roots]
        excluded = exclusion_mask_for_samples(
            self.alt_rgb_exclusions,
            dataset_names,
            self.split,
            self.sample_sequence_names,
            self.sample_image_indices,
            frame_names,
        )
        if excluded is None:
            return None
        known_datasets = set(self.alt_rgb_exclusions.datasets)
        matched = [name for name in dataset_names if name in known_datasets]
        if not matched:
            print(
                f"[warn] alt RGB exclusions loaded from {self.alt_rgb_exclude_json} match none of "
                f"{sorted(known_datasets)}; every sample keeps its inpainted RGB"
            )
        else:
            print(
                f"[info] alt RGB exclusions: {int(excluded.sum())} sample/root pairs fall back to the "
                f"original RGB (datasets={sorted(matched)}, split={self.split})"
            )
        return excluded

    def _avoid_flagged_alt_rgb(self, rng: np.random.Generator):
        """Never use an inpainted frame that RC-S flagged for that dataset.

        The sampled root is replaced by one that did not flag this frame (same frame, so the
        RGB still matches the events and labels); frames flagged by every root keep the
        original RGB.
        """
        self.use_alt_rgb_for_sample, self.alt_rgb_root_index_for_sample = reroute_flagged_samples(
            self.use_alt_rgb_for_sample,
            self.alt_rgb_root_index_for_sample,
            self.alt_rgb_excluded_per_root,
            rng,
        )

    def rgb_frame_names(self, sequence_name: str) -> List[str]:
        """Frame file names of the RGB source that ``__getitem__`` uses for one sequence."""
        return [Path(path).name for path in self.primary_rgb_image_files[sequence_name]]

    def drop_excluded_frames(self, exclusions: Optional[RgbExclusions], dataset_name: Optional[str] = None,
                             use_union: bool = False) -> Dict[str, int]:
        """Remove samples whose first frame was flagged by RC-S.

        Used by the evaluation scripts so that every RGB variant is scored on the same
        frames. ``dataset_name`` selects the exclusion list of one DSEC-Emerge set;
        ``use_union`` drops frames flagged for *any* dataset instead. Returns the number of
        dropped samples per sequence.
        """
        if exclusions is None or not exclusions:
            return {}
        if not use_union and dataset_name is None:
            raise ValueError("drop_excluded_frames needs dataset_name or use_union=True")
        dropped: Dict[str, int] = {}
        for sequence_name, pairs in list(self.image_index_pairs.items()):
            if pairs is None or len(pairs) == 0:
                continue
            frame_names = self.rgb_frame_names(sequence_name)
            keep = np.ones(len(pairs), dtype=bool)
            for index in range(len(pairs)):
                frame_name = frame_names[int(pairs[index][0])]
                if use_union:
                    flagged = exclusions.contains_any(self.split, sequence_name, frame_name)
                else:
                    flagged = exclusions.contains(dataset_name, self.split, sequence_name, frame_name)
                if flagged:
                    keep[index] = False
            dropped_here = int((~keep).sum())
            if dropped_here:
                self.image_index_pairs[sequence_name] = pairs[keep]
                dropped[sequence_name] = dropped_here
        if dropped:
            self.sample_sequence_names, self.sample_image_indices = self._build_sample_index()
            self.alt_rgb_excluded_per_root = (
                None if self.alt_rgb_excluded_per_root is None else self._build_alt_rgb_excluded_per_root()
            )
            self.resample_image_sources(epoch=0)
            self.resample_interframe_offsets(epoch=0)
        return dropped

    def resample_interframe_offsets(self, epoch: int):
        choices = self.interframe_ratios if self.interframe_ratios is not None else self.interframe_offsets_us
        if choices is None:
            self.sample_interframe_offsets_us = None
            self.sample_interframe_ratios = None
            return

        if self.randomize_interframe_offsets:
            rng = np.random.default_rng(self.interframe_offset_seed + int(epoch))
            sampled = rng.choice(choices, size=len(self.sample_sequence_names), replace=True)
        else:
            repeats = int(np.ceil(len(self.sample_sequence_names) / len(choices)))
            sampled = np.tile(choices, repeats)[:len(self.sample_sequence_names)]

        if self.interframe_ratios is not None:
            self.sample_interframe_ratios = sampled.astype(np.float64, copy=False)
            self.sample_interframe_offsets_us = None
        else:
            self.sample_interframe_offsets_us = sampled.astype(np.int64, copy=False)
            self.sample_interframe_ratios = None

    def _load_rgb_image(self, sample_index: int, sequence_name: str, image_index: int):
        if self.use_alt_rgb_for_sample[sample_index]:
            alt_root_index = int(self.alt_rgb_root_index_for_sample[sample_index])
            image_path = self.alt_rgb_image_files[sequence_name][alt_root_index][image_index]
        else:
            image_path = self.primary_rgb_image_files[sequence_name][image_index]
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Failed to load RGB image for sequence={sequence_name}, image_index={image_index}, path={image_path}"
            )
        return image

    def _removed_track_ids_for_sample(self, sample_index: int, sequence_name: str, image_index: int):
        if not self.removed_targets_only_for_alt or not self.use_alt_rgb_for_sample[sample_index]:
            return None
        return self.removed_track_ids_for_sample(sample_index, sequence_name, image_index)

    def removed_track_ids_for_sample(self, sample_index: int, sequence_name: str, image_index: int):
        """Removed target ids of this sample's alternate RGB frame.

        Unlike ``_removed_track_ids_for_sample`` this is *not* gated by
        ``removed_targets_only_for_alt`` and does not cause any GT filtering; it is
        the read-only accessor used by the emergence mask / metric plumbing.
        Returns ``None`` when the sample does not use an inpainted RGB.
        """
        if not self.use_alt_rgb_for_sample[sample_index]:
            return None
        alt_root_index = int(self.alt_rgb_root_index_for_sample[sample_index])
        image_path = self.alt_rgb_image_files[sequence_name][alt_root_index][image_index]
        frame_name = image_path.name
        return self.alt_removed_track_ids[alt_root_index].get(sequence_name, {}).get(frame_name, set())

    def set_num_us(self, num_us):
        self.fixed_interframe_ratio = None
        self.sample_interframe_offsets_us = None
        self.sample_interframe_ratios = None
        self.num_us = num_us

    def set_interframe_ratio(self, ratio: float):
        ratio = float(ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"interframe ratio must be in (0, 1], got {ratio}")
        self.fixed_interframe_ratio = ratio

    def _resolve_interframe_offset_us(self, sample_index: int, frame_interval_us: int):
        fixed_ratio = getattr(self, "fixed_interframe_ratio", None)
        if fixed_ratio is not None:
            return int(round(fixed_ratio * frame_interval_us))
        if self.sample_interframe_ratios is not None:
            return int(round(float(self.sample_interframe_ratios[sample_index]) * frame_interval_us))
        if self.sample_interframe_offsets_us is not None:
            return int(self.sample_interframe_offsets_us[sample_index])
        if self.num_us >= 0:
            return int(self.num_us)
        return None

    def visualize_debug(self, index):
        data = self.__getitem__(index)
        image = data.image[0].permute(1,2,0).numpy()
        p = data.x[:,0].numpy()
        x, y = data.pos.t().numpy()
        b_x, b_y, b_w, b_h, b_c = data.bbox.t().numpy()

        image = draw_events_on_image(image, x, y, p)
        image = draw_bbox_on_img(image, b_x, b_y, b_w, b_h,
                                 b_c, np.ones_like(b_c), conf=0.3, nms=0.65)

        cv2.imshow(f"Debug {index}", image)
        cv2.waitKey(0)


    def __len__(self):
        return sum(len(d) for d in self.image_index_pairs.values())

    def preprocess_detections(self, detections):
        detections = rescale_tracks(detections, self.scale)
        detections = crop_tracks(detections, self.width, self.height)
        detections['class_id'], _ = map_classes(detections['class_id'], self.class_remapping)
        return detections

    def preprocess_events(self, events):
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
        events['p'] = 2 * events['p'].reshape((-1,1)).astype("int8") - 1
        return events

    def preprocess_image(self, image):
        image = image[:self.scale * self.height]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, idx):
        sample_index = idx
        dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
        image_index_0, image_index_1 = image_index_pairs[idx]
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]
        frame_interval_us = int(image_ts_1 - image_ts_0)

        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
        detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

        detections_0 = self.preprocess_detections(detections_0)
        detections_1 = self.preprocess_detections(detections_1)

        image_0 = self._load_rgb_image(sample_index, dataset.root.name, image_index_0)
        image_0 = self.preprocess_image(image_0)

        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        offset_us = self._resolve_interframe_offset_us(sample_index, frame_interval_us)
        if offset_us is not None:
            image_ts_1 = image_ts_0 + offset_us
            events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
            if not self.no_eval:
                if self.shared_track_interpolation:
                    detections_1 = interpolate_shared_tracks(detections_0, detections_1, image_ts_1)
                else:
                    detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)

        removed_track_ids = self._removed_track_ids_for_sample(sample_index, dataset.root.name, image_index_0)
        if removed_track_ids is not None:
            detections_0 = filter_tracks_by_track_ids(detections_0, removed_track_ids)
            detections_1 = filter_tracks_by_track_ids(detections_1, removed_track_ids)

        # here, the timestamp of the events is no longer absolute
        events = self.preprocess_events(events)

        # convert to torch geometric data
        data = to_data(**events, bbox=tracks_to_array(detections_1), bbox0=tracks_to_array(detections_0),
                       track_ids0=torch.from_numpy(detections_0["track_id"].astype(np.int64)),
                       track_ids1=torch.from_numpy(detections_1["track_id"].astype(np.int64)),
                       t0=image_ts_0, t1=image_ts_1,
                       width=self.width, height=self.height, time_window=self.time_window,
                       image=image_0, sequence=str(dataset.root.name))

        if self.transform is not None:
            data = self.transform(data)

        # remove bboxes if they have 0 width or height
        mask = filter_small_bboxes(data.bbox[:, 2], data.bbox[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox = data.bbox[mask]
        data.track_ids1 = data.track_ids1[mask]
        mask = filter_small_bboxes(data.bbox0[:, 2], data.bbox0[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox0 = data.bbox0[mask]
        data.track_ids0 = data.track_ids0[mask]

        # Inter-frame offset actually used for this sample (offset when configured,
        # otherwise the full RGB frame interval). Consumed by the reliability-gated
        # fusion variant as the normalized Δt condition.
        data.delta_t_us = torch.tensor([float(image_ts_1 - image_ts_0)], dtype=torch.float32)

        # Per-box flags marking the targets whose RGB evidence was removed (DSEC-Emerge).
        # ``removed_mask0`` is aligned with ``bbox0``/``track_ids0`` (t0),
        # ``removed_mask1`` with ``bbox``/``track_ids1`` (t1). Both are all-False on
        # samples that use the original RGB.
        removed_ids = self.removed_track_ids_for_sample(sample_index, dataset.root.name, image_index_0)
        if removed_ids:
            removed_tensor = torch.tensor(sorted(int(t) for t in removed_ids), dtype=data.track_ids0.dtype)
            data.removed_mask0 = torch.isin(data.track_ids0, removed_tensor)
            data.removed_mask1 = torch.isin(data.track_ids1, removed_tensor)
        else:
            data.removed_mask0 = torch.zeros_like(data.track_ids0, dtype=torch.bool)
            data.removed_mask1 = torch.zeros_like(data.track_ids1, dtype=torch.bool)

        # Rasterize the t0 boxes of removed targets for emergence-mask supervision.
        # RGB evidence is erased at t0 and the fused features are t0-aligned, so the mask lives
        # on the t0 grid; because it is built from the *already transformed* ``bbox0`` it
        # inherits the geometric augmentation for free (no ``augment.py`` change needed).
        # ``emergence_valid`` separates "this sample carries usable supervision" from the three
        # no-signal cases (original RGB / ``copied`` frame / nothing survived the size filter),
        # which would otherwise all look like an all-zero mask.
        data.emergence_mask = torch.zeros(1, self.height, self.width, dtype=torch.bool)
        data.emergence_valid = torch.zeros(1)
        if self.emergence_mask_enabled and bool(data.removed_mask0.any()):
            for x, y, w, h in data.bbox0[data.removed_mask0, :4].tolist():
                x0 = max(int(np.floor(x)), 0)
                y0 = max(int(np.floor(y)), 0)
                x1 = min(int(np.ceil(x + w)), self.width)
                y1 = min(int(np.ceil(y + h)), self.height)
                if x1 > x0 and y1 > y0:
                    data.emergence_mask[0, y0:y1, x0:x1] = True
            if bool(data.emergence_mask.any()):
                data.emergence_valid = torch.ones(1)

        return data

    def rel_index(self, idx):
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            directory = self.dataset.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):
                return directory, image_index_pairs, track_mask, idx
            idx -= len(image_index_pairs)
        raise IndexError
