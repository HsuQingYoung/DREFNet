"""Paired original/emergence recall on the same removed-object targets.

For every DSEC-Emerge (inpainted) sample, takes the ground-truth boxes of the
targets whose RGB evidence was removed, and runs the detector twice: once with
the original RGB frame and once with the inpainted RGB frame. Both predictions
are scored against the same removed-object GT subset. The recall on the other
objects of the inpainted frame is retained as a secondary diagnostic.

Usage:
    python3 scripts/eval_emergence_recall.py \
        --onnx weights/DREFNet.onnx \
        --output_directory outputs/recall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch_geometric.data import DataLoader
from torchvision.ops import box_iou

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "runtime"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drefnet import EventPillarsEncoder
from drefnet.data.augment import Augmentations
from drefnet.data.dsec_data import DSEC
from drefnet.data.rgb_exclusions import RgbExclusions, dataset_name_for_root
from drefnet.model_utils import (
    postprocess_network_output,
)

FOLLOW_BATCH_KEYS = ["bbox", "bbox0"]


class OnnxDetector:
    def __init__(self, model_path: Path, provider: str):
        available = ort.get_available_providers()
        self.provider = "CUDAExecutionProvider" if provider == "auto" and "CUDAExecutionProvider" in available else provider
        if self.provider == "auto":
            self.provider = "CPUExecutionProvider"
        if self.provider not in available:
            raise RuntimeError(f"Provider {self.provider} is unavailable; installed providers: {available}")
        self.session = ort.InferenceSession(str(model_path), providers=[self.provider])
        self.input_names = {item.name for item in self.session.get_inputs()}
        if self.input_names != {"rgb_frame", "event_pillars"}:
            raise ValueError(f"Unexpected ONNX inputs: {sorted(self.input_names)}")
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}

    def __call__(self, inputs):
        feeds = {
            "rgb_frame": inputs["rgb_frame"].detach().cpu().numpy(),
            "event_pillars": inputs["event_pillars"].detach().cpu().numpy(),
        }
        for name, value in feeds.items():
            if list(value.shape) != self.input_shapes[name]:
                raise ValueError(f"{name} shape {value.shape} does not match ONNX input {self.input_shapes[name]}")
        output = self.session.run(["predictions"], feeds)[0]
        return torch.from_numpy(output).to(inputs["rgb_frame"].device)


def build_eventpillars_batch(data, encoder, device):
    pillars = []
    for b in range(data.num_graphs):
        mask = data.batch == b
        events = torch.cat(
            [data.pos[mask].float(), data.t[mask].float().unsqueeze(1), data.x[mask].float()], dim=1
        )
        pillars.append(encoder.encode(events, device=device))
    return torch.stack(pillars, dim=0)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes[:, :4].clone()
    out[:, 2:4] += out[:, :2]
    return out


def recall_on(gt_boxes, gt_labels, det_boxes, det_labels, iou_thre):
    """(matched, total) for the given GT subset, matching by class + IoU."""
    total = len(gt_boxes)
    if total == 0:
        return 0, 0
    if len(det_boxes) == 0:
        return 0, total
    ious = box_iou(gt_boxes, det_boxes)
    matched = 0
    for g in range(total):
        same_class = det_labels == gt_labels[g]
        if bool(same_class.any()) and float(ious[g][same_class].max()) >= iou_thre:
            matched += 1
    return matched, total


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--provider", choices=["auto", "CPUExecutionProvider", "CUDAExecutionProvider"], default="auto")
    parser.add_argument("--dataset_directory", type=Path, required=True)
    parser.add_argument("--alt_rgb_dataset_directory", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude_json", type=Path, default=None,
                        help="optional external frame-exclusion JSON; applied per_dataset, as in the mAP evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--only_perfect_tracks", type=int, default=1, choices=[0, 1],
                        help="1 (default, legacy): keep only frame pairs whose track set is identical and whose "
                             "min IoU >= 0.10; 0: keep all frames and interpolate the shared tracks only "
                             "(matches the stage-2 training / validation protocol)")
    parser.add_argument("--shared_track_interpolation", type=int, default=0, choices=[0, 1],
                        help="interpolate only tracks present in both frames (required when --only_perfect_tracks 0)")
    parser.add_argument("--offsets", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--conf_thre", type=float, default=0.3)
    parser.add_argument("--nms_thre", type=float, default=0.65)
    parser.add_argument("--iou_thre", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="max samples per (dataset, offset); 0 (default) = all")
    parser.add_argument("--include_original", type=int, default=1, choices=[0, 1],
                        help="deprecated compatibility flag; paired original recall is always computed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_directory", type=Path, default=_HERE.parents[1] / "outputs/eval_results/figure")
    parser.add_argument("--preview_png", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.batch_size != 1:
        raise ValueError("This ONNX model has a fixed batch size of 1; use --batch_size 1")
    net = OnnxDetector(args.onnx.expanduser(), args.provider)
    print(f"onnx={args.onnx} num_classes=2 provider={net.provider}")

    exclusions = RgbExclusions.from_json(args.exclude_json) if args.exclude_json else None

    results = {}  # emergence dataset name -> offset -> dict
    entries = []
    for root in args.alt_rgb_dataset_directory:
        root = root.expanduser()
        if root.name != "dsec":
            root = root / "dsec"
        entries.append((dataset_name_for_root(root), root))

    for name, root in entries:
        common_dataset_kwargs = dict(
            root=args.dataset_directory.expanduser(),
            split=args.split,
            transform=Augmentations.transform_testing,
            debug=False,
            min_bbox_diag=15,
            min_bbox_height=10,
            only_perfect_tracks=bool(args.only_perfect_tracks),
            shared_track_interpolation=bool(args.shared_track_interpolation) or not bool(args.only_perfect_tracks),
            no_eval=False,
        )
        # These two datasets share the DSEC annotations/events and sample order.  The
        # only intended difference is the RGB source, which makes the comparison
        # paired rather than comparing different GT subsets.
        original_dataset = DSEC(**common_dataset_kwargs)
        emergence_dataset = DSEC(
            **common_dataset_kwargs,
            alt_rgb_roots=[root],
            alt_rgb_sampling_ratio=1.0,
        )
        if exclusions is not None:
            dropped_original = original_dataset.drop_excluded_frames(exclusions, dataset_name=name)
            dropped_emergence = emergence_dataset.drop_excluded_frames(exclusions, dataset_name=name)
            if dropped_original != dropped_emergence:
                raise RuntimeError(
                    f"Original/emergence frame filtering differs for {name}: "
                    f"original={dropped_original}, emergence={dropped_emergence}"
                )
            print(f"[exclude] {name}: dropped {sum(dropped_emergence.values())} frames, "
                  f"{len(emergence_dataset)} left")

        if len(original_dataset) != len(emergence_dataset):
            raise RuntimeError(
                f"Original/emergence dataset lengths differ for {name}: "
                f"{len(original_dataset)} vs {len(emergence_dataset)}"
            )

        encoder = EventPillarsEncoder(height=emergence_dataset.height, width=emergence_dataset.width)
        results[name] = {}
        for offset in args.offsets:
            original_dataset.set_num_us(int(offset * 1000))
            emergence_dataset.set_num_us(int(offset * 1000))
            original_loader = DataLoader(
                original_dataset, follow_batch=FOLLOW_BATCH_KEYS, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, drop_last=False,
            )
            emergence_loader = DataLoader(
                emergence_dataset, follow_batch=FOLLOW_BATCH_KEYS, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers, drop_last=False,
            )
            agg = dict(
                original_matched=0, original_total=0,
                emergence_matched=0, emergence_total=0,
                oth_matched=0, oth_total=0, samples=0, skipped=0,
            )
            seen = 0
            for original_data, emergence_data in zip(original_loader, emergence_loader):
                if args.max_samples and seen >= args.max_samples:
                    break
                if original_data.num_graphs != emergence_data.num_graphs:
                    raise RuntimeError(f"Paired batch size differs for {name} @ {offset}ms")
                original_timestamps = torch.as_tensor([original_data.t0, original_data.t1])
                emergence_timestamps = torch.as_tensor([emergence_data.t0, emergence_data.t1])
                if not torch.equal(original_timestamps, emergence_timestamps):
                    raise RuntimeError(f"Paired timestamps differ for {name} @ {offset}ms")
                if (
                    not torch.equal(original_data.bbox.cpu(), emergence_data.bbox.cpu())
                    or not torch.equal(original_data.track_ids1.cpu(), emergence_data.track_ids1.cpu())
                    or not torch.equal(original_data.bbox_batch.cpu(), emergence_data.bbox_batch.cpu())
                ):
                    raise RuntimeError(f"Paired GT differs for {name} @ {offset}ms")
                seen += emergence_data.num_graphs
                original_data = original_data.to(device)
                emergence_data = emergence_data.to(device)
                original_image = original_data.image.float() / 255.0
                emergence_image = emergence_data.image.float() / 255.0
                if original_image.ndim == 5 and original_image.shape[1] == 1:
                    original_image = original_image[:, 0]
                    emergence_image = emergence_image[:, 0]
                event_pillars = build_eventpillars_batch(emergence_data, encoder, device)
                dt_norm = torch.full((emergence_data.num_graphs,), offset / 50.0, device=device)
                original_pred = net({
                    "rgb_frame": original_image,
                    "event_pillars": event_pillars,
                    "reuse_rgb": False,
                    "delta_t_norm": dt_norm,
                })
                emergence_pred = net({
                    "rgb_frame": emergence_image,
                    "event_pillars": event_pillars,
                    "reuse_rgb": False,
                    "delta_t_norm": dt_norm,
                })
                original_detections = postprocess_network_output(
                    original_pred, num_classes=2, conf_thre=args.conf_thre,
                    nms_thre=args.nms_thre, height=emergence_dataset.height,
                    width=emergence_dataset.width, filtering=True,
                )
                emergence_detections = postprocess_network_output(
                    emergence_pred, num_classes=2, conf_thre=args.conf_thre,
                    nms_thre=args.nms_thre, height=emergence_dataset.height,
                    width=emergence_dataset.width, filtering=True,
                )
                for b in range(emergence_data.num_graphs):
                    row = emergence_data.bbox_batch == b
                    if not bool(row.any()):
                        continue
                    boxes = emergence_data.bbox[row]
                    removed = emergence_data.removed_mask1[row]
                    if not bool(removed.any()):
                        agg["skipped"] += 1
                        continue
                    gt_boxes = xywh_to_xyxy(boxes).cpu()
                    gt_labels = boxes[:, 4].long().cpu()
                    removed = removed.cpu()
                    original_det = original_detections[b]
                    emergence_det = emergence_detections[b]
                    m, t = recall_on(
                        gt_boxes[removed], gt_labels[removed], original_det["boxes"].cpu(),
                        original_det["labels"].cpu(), args.iou_thre,
                    )
                    agg["original_matched"] += m
                    agg["original_total"] += t
                    m, t = recall_on(
                        gt_boxes[removed], gt_labels[removed], emergence_det["boxes"].cpu(),
                        emergence_det["labels"].cpu(), args.iou_thre,
                    )
                    agg["emergence_matched"] += m
                    agg["emergence_total"] += t
                    m, t = recall_on(
                        gt_boxes[~removed], gt_labels[~removed], emergence_det["boxes"].cpu(),
                        emergence_det["labels"].cpu(), args.iou_thre,
                    )
                    agg["oth_matched"] += m
                    agg["oth_total"] += t
                    agg["samples"] += 1
            original_recall = (
                agg["original_matched"] / agg["original_total"] if agg["original_total"] else float("nan")
            )
            emergence_recall = (
                agg["emergence_matched"] / agg["emergence_total"] if agg["emergence_total"] else float("nan")
            )
            oth = agg["oth_matched"] / agg["oth_total"] if agg["oth_total"] else float("nan")
            recall_drop = original_recall - emergence_recall
            relative_recall = emergence_recall / original_recall if original_recall else float("nan")
            results[name][offset] = dict(
                original_recall=original_recall, emergence_recall=emergence_recall,
                recall_drop=recall_drop, relative_recall=relative_recall,
                oth=oth, gap=oth - emergence_recall, **agg,
            )
            print(
                f"{name} @ {offset:>2d}ms  original_recall={original_recall:.4f} "
                f"({agg['original_matched']}/{agg['original_total']})  "
                f"emergence_recall={emergence_recall:.4f} ({agg['emergence_matched']}/"
                f"{agg['emergence_total']})  drop={recall_drop:+.4f}  "
                f"other_recall={oth:.4f}  samples={agg['samples']} skipped={agg['skipped']}"
            )

    # ---- table ----
    print("\n=== Emergence Recall (IoU>=%.2f, conf>=%.2f) ===" % (args.iou_thre, args.conf_thre))
    for name, per_offset in results.items():
        print(f"\n[{name}]")
        print(f"{'Δt(ms)':>7s}{'original':>11s}{'emergence':>11s}{'drop':>9s}{'#removed':>10s}{'#samples':>10s}")
        for offset in args.offsets:
            r = per_offset[offset]
            print(f"{offset:7d}{r['original_recall']:11.4f}{r['emergence_recall']:11.4f}"
                  f"{r['recall_drop']:+9.4f}{r['emergence_total']:10d}{r['samples']:10d}")
        mean_original = np.nanmean([per_offset[o]["original_recall"] for o in args.offsets])
        mean_emergence = np.nanmean([per_offset[o]["emergence_recall"] for o in args.offsets])
        print(f"{'mean':>7s}{mean_original:11.4f}{mean_emergence:11.4f}"
              f"{mean_original - mean_emergence:+9.4f}")

    # ---- figure ----
    names = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0.05, 0.8, len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for color, name in zip(colors, names):
        original = [results[name][o]["original_recall"] for o in args.offsets]
        emergence = [results[name][o]["emergence_recall"] for o in args.offsets]
        drop = [results[name][o]["recall_drop"] for o in args.offsets]
        axes[0].plot(args.offsets, original, marker="o", color=color, linewidth=1.9, label=f"{name}: original")
        axes[0].plot(args.offsets, emergence, marker="s", linestyle="--", color=color, linewidth=1.7,
                     label=f"{name}: emergence")
        axes[1].plot(args.offsets, drop, marker="s", color=color, linewidth=1.9, label=name)
    axes[0].set_xlabel(r"Event window $\Delta t$ (ms)")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Paired recall on removed objects", fontsize=11)
    axes[1].set_xlabel(r"Event window $\Delta t$ (ms)")
    axes[1].set_ylabel("original recall − emergence recall")
    axes[1].set_title("Recall drop from RGB removal", fontsize=11)
    axes[1].axhline(0.0, color="gray", linewidth=1.0, linestyle=":")
    for ax in axes:
        ax.set_xticks(list(range(5, 51, 5)))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Emergence Recall  |  IoU≥{args.iou_thre}, conf≥{args.conf_thre}", fontsize=12, y=1.02)
    fig.tight_layout()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.only_perfect_tracks else "_allframes"
    svg = args.output_directory / f"emergence_recall{suffix}.svg"
    fig.savefig(svg, bbox_inches="tight")
    print(f"\nsaved: {svg}")
    if args.preview_png is not None:
        fig.savefig(args.preview_png, dpi=150, bbox_inches="tight")
        print(f"saved (preview): {args.preview_png}")

    csv = args.output_directory / f"emergence_recall{suffix}.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("dataset,dt_ms,original_recall,emergence_recall,recall_drop,relative_recall,"
                "other_recall,other_gap,n_original_matched,n_original_total,n_emergence_matched,"
                "n_emergence_total,n_other_matched,n_other_total,n_samples\n")
        for name, per_offset in results.items():
            for offset in args.offsets:
                r = per_offset[offset]
                f.write(f"{name},{offset},{r['original_recall']:.6f},{r['emergence_recall']:.6f},"
                        f"{r['recall_drop']:.6f},{r['relative_recall']:.6f},{r['oth']:.6f},"
                        f"{r['gap']:.6f},{r['original_matched']},{r['original_total']},"
                        f"{r['emergence_matched']},{r['emergence_total']},{r['oth_matched']},"
                        f"{r['oth_total']},{r['samples']}\n")
    print(f"saved: {csv}")


if __name__ == "__main__":
    main()
