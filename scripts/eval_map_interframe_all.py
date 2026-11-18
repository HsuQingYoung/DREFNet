"""Evaluate interframe mAP across original and alternate DSEC RGB roots."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch_geometric.data import DataLoader

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "runtime"))

from drefnet import EventPillarsEncoder
from drefnet.model_utils import eval_report_label
from drefnet.data.augment import Augmentations
from drefnet.data.dsec_data import DSEC
from drefnet.data.rgb_exclusions import RgbExclusions, dataset_name_for_root
from drefnet.model_utils import convert_to_evaluation_format, postprocess_network_output
from drefnet.utils.buffers import DetectionBuffer


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


def resolve_dsec_root(path: Path) -> Path:
    path = path.expanduser()
    return path if path.name == "dsec" else path / "dsec"


def build_eventpillars_batch(data, encoder, device):
    pillars = []
    for batch_index in range(data.num_graphs):
        mask = data.batch == batch_index
        events = torch.cat(
            [data.pos[mask].float(), data.t[mask].float().unsqueeze(1), data.x[mask].float()], dim=1
        )
        pillars.append(encoder.encode(events, device=device))
    return torch.stack(pillars, dim=0)


def build_delta_t_norm(data, device):
    delta_t_us = getattr(data, "delta_t_us", None)
    if delta_t_us is None:
        return torch.ones(data.num_graphs, device=device)
    return (delta_t_us.to(device).float() / 50000.0).clamp(0.0, 1.0)


def infer_frame_interval_us(dataset) -> int:
    directory, image_index_pairs, _, relative_index = dataset.rel_index(0)
    image_index_0, image_index_1 = image_index_pairs[relative_index]
    timestamps = directory.images.timestamps[[image_index_0, image_index_1]]
    interval_us = int(timestamps[1] - timestamps[0])
    if interval_us <= 0:
        raise ValueError(f"Invalid frame interval: {interval_us}")
    return interval_us


def build_eval_offsets(frame_interval_us: int, start_step: int, num_interframe_steps: int):
    if num_interframe_steps <= 0 or not 1 <= start_step <= num_interframe_steps:
        raise ValueError("Require num_interframe_steps > 0 and 1 <= start_step <= num_interframe_steps")
    offsets = [
        int(round(value))
        for value in np.linspace(0, frame_interval_us, num_interframe_steps + 1)[1:][start_step - 1:]
    ]
    return list(dict.fromkeys(offsets))


@torch.no_grad()
def evaluate_at_offset(dataset, loader, net, encoder, device, cfg, args, offset_us):
    dataset.set_num_us(offset_us)
    metric = DetectionBuffer(height=dataset.height, width=dataset.width, classes=dataset.classes)
    for batch_index, data in enumerate(loader):
        data = data.to(device)
        image = data.image.float() / 255.0
        if image.ndim == 5 and image.shape[1] == 1:
            image = image[:, 0]
        inputs = {
            "rgb_frame": image,
            "event_pillars": build_eventpillars_batch(data, encoder, device),
            "reuse_rgb": False,
            "delta_t_norm": build_delta_t_norm(data, device),
        }
        predictions = net(inputs)
        detections = postprocess_network_output(
            predictions, num_classes=2, conf_thre=args.conf_thre,
            nms_thre=args.nms_thre, height=dataset.height, width=dataset.width, filtering=True,
        )
        targets = convert_to_evaluation_format(data)
        metric.update(detections, targets, dataset="dsec", height=dataset.height, width=dataset.width)
        if (batch_index + 1) % 50 == 0:
            print(f"offset_us={offset_us}: processed batches {batch_index + 1}/{len(loader)}")
    return {key: float(value) for key, value in metric.compute().items()}, {}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate interframe mAP on all RGB variants")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--provider", choices=["auto", "CPUExecutionProvider", "CUDAExecutionProvider"], default="auto")
    parser.add_argument("--dataset_directory", type=Path, required=True)
    parser.add_argument("--alt_rgb_dataset_directory", type=Path, nargs="*", default=[])
    parser.add_argument("--output_directory", type=Path, default=Path("outputs/eval_results"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--only_perfect_tracks", type=int, default=1, choices=[0, 1],
                        help="1 (default, legacy): keep only frame pairs whose track set is identical and whose "
                             "min IoU >= 0.10; 0: keep all frames and interpolate the shared tracks only "
                             "(matches the stage-2 training / validation protocol)")
    parser.add_argument("--shared_track_interpolation", type=int, default=0, choices=[0, 1],
                        help="interpolate only tracks present in both frames (required when --only_perfect_tracks 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_json",
        type=Path,
        default=None,
        help="optional external frame-exclusion JSON; flagged frames are dropped from the evaluation",
    )
    parser.add_argument(
        "--exclude_scope",
        type=str,
        default="per_dataset",
        choices=["per_dataset", "union"],
        help="per_dataset: each mask set is scored on its own kept frames (the 'original' row stays full); "
             "union: every row (including 'original') is scored on the frames that no mask set flagged",
    )
    parser.add_argument("--model_variant", type=str, default="crgf", choices=["crgf"])
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_ss2d", "no_gate"])
    parser.add_argument("--event_representation", choices=["eventpillars"], default="eventpillars")
    parser.add_argument("--neck_type", choices=["pafpn"], default="pafpn")
    parser.add_argument("--reliability_gate_mode", type=str, default="spatial_only", choices=["spatial_only", "no_ss2d", "none"])
    parser.add_argument("--reliability_gate_stages", type=str, default=None,
                        help="unused compatibility option")
    parser.add_argument("--num_interframe_steps", type=int, default=10)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--conf_thre", type=float, default=0.001)
    parser.add_argument("--nms_thre", type=float, default=0.65)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101"],
                        help="Legacy default for both branches; branch-specific options take precedence.")
    parser.add_argument("--rgb_backbone", type=str, default="resnet50", choices=["resnet50"])
    parser.add_argument("--event_backbone", type=str, default="resnet34", choices=["resnet34"])
    parser.add_argument(
        "--event_native_channels",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=事件支路直接使用原生通道（默认）；0=通过 adapter 升维到 RGB 通道（旧行为）",
    )
    return parser.parse_args()


def write_markdown_report(path, args, dataset_path, results):
    metric_names = sorted(
        {
            metric
            for per_step, _ in results.values()
            for item in per_step
            for metric in item["metrics"]
        }
    )
    lines = [
        "# Interframe mAP Evaluation",
        "",
        f"- ONNX model: `{args.onnx}`",
        f"- Dataset: `{dataset_path}`",
        f"- Split: `{args.split}`",
        f"- Model variant: `{args.model_variant}`",
        f"- RGB backbone: `{getattr(args, 'rgb_backbone', None) or getattr(args, 'backbone', 'resnet50')}`",
        f"- Event backbone: `{getattr(args, 'event_backbone', None) or getattr(args, 'backbone', 'resnet50')}`",
        f"- Exclude json: `{getattr(args, 'exclude_json', None)}` (scope: `{getattr(args, 'exclude_scope', None)}`)",
        "",
        "| Dataset | Step | Offset (us) | " + " | ".join(metric_names) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in metric_names) + " |",
    ]
    for name, (per_step, _) in results.items():
        for item in per_step:
            values = " | ".join(f"{item['metrics'].get(metric, float('nan')):.6f}" for metric in metric_names)
            lines.append(
                f"| {name} | {item['step_idx']} | {item['offset_us']} | {values} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_dataset(name, dataset, net, encoder, device, cfg, args, offsets, frame_interval_us):
    loader = DataLoader(dataset, follow_batch=["bbox", "bbox0"], batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    per_step = []
    for step_idx, offset_us in enumerate(offsets, start=args.start_step):
        metrics, _ = evaluate_at_offset(
            dataset, loader, net, encoder, device, cfg, args, offset_us
        )
        per_step.append({"step_idx": step_idx, "offset_us": offset_us, "metrics": metrics})
    for item in per_step:
        print(f"STEP [{name}] {item['step_idx']}: offset_us={item['offset_us']}, metrics={item['metrics']}")
    return per_step, {}


@torch.no_grad()
def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This ONNX model has a fixed batch size of 1; use --batch_size 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = resolve_dsec_root(args.dataset_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"dataset(events/gt)={dataset_path}")
    print(f"onnx={args.onnx}")
    print(f"event_representation={args.event_representation}")

    dataset_kwargs = dict(root=dataset_path, split=args.split, transform=Augmentations.transform_testing,
                          debug=False, min_bbox_diag=15, min_bbox_height=10,
                          only_perfect_tracks=bool(args.only_perfect_tracks),
                          shared_track_interpolation=bool(args.shared_track_interpolation) or not bool(args.only_perfect_tracks),
                          no_eval=False)
    dataset = DSEC(**dataset_kwargs)
    exclusions = RgbExclusions.from_json(args.exclude_json) if args.exclude_json else None
    if exclusions is not None:
        print(f"exclude.json={args.exclude_json} ({len(exclusions)} flagged frames, scope={args.exclude_scope})")
        if args.exclude_scope == "union":
            dropped = sum(dataset.drop_excluded_frames(exclusions, use_union=True).values())
            print(f"[exclude] original: dropped {dropped} frames, {len(dataset)} left")
    rgb_backbone = "resnet50"
    event_backbone = "resnet34"
    print(f"rgb_backbone={rgb_backbone}, event_backbone={event_backbone}, model_variant={args.model_variant}")
    args.rgb_backbone = rgb_backbone
    args.event_backbone = event_backbone
    net = OnnxDetector(args.onnx.expanduser(), args.provider)
    cfg = argparse.Namespace(num_classes=2)
    print(f"provider={net.provider} num_classes=2")
    encoder = EventPillarsEncoder(height=dataset.height, width=dataset.width)
    datasets = {"original": dataset}
    for rgb_root in args.alt_rgb_dataset_directory:
        rgb_root = resolve_dsec_root(rgb_root)
        name = dataset_name_for_root(rgb_root)
        masked = DSEC(
            **dataset_kwargs,
            alt_rgb_roots=[rgb_root],
            rgb_sampling_weights=[0.0, 1.0],
        )
        if exclusions is not None:
            dropped_here = masked.drop_excluded_frames(
                exclusions, dataset_name=name, use_union=args.exclude_scope == "union"
            )
            print(f"[exclude] {name}: dropped {sum(dropped_here.values())} frames, {len(masked)} left")
        datasets[name] = masked
    results = {}
    frames = {name: len(current) for name, current in datasets.items()}
    for name, current_dataset in datasets.items():
        current_interval = infer_frame_interval_us(current_dataset)
        current_offsets = build_eval_offsets(current_interval, args.start_step, args.num_interframe_steps)
        results[name] = evaluate_dataset(name, current_dataset, net, encoder, device, cfg, args,
                                         current_offsets, current_interval)
    print("INTERFRAME_ALL_SUMMARY:", results)
    model_label = eval_report_label(args.model_variant, rgb_backbone, event_backbone, args.onnx)
    report_path = args.output_directory.expanduser() / f"{model_label}_interframe{'' if args.only_perfect_tracks else '_allframes'}_mAP.md"
    write_markdown_report(report_path, args, dataset_path, results)
    print(f"[frames] {frames}")
    print(f"Markdown results written to {report_path}")


if __name__ == "__main__":
    main()
