from pathlib import Path

import torch
import torchvision


def eval_report_label(model_variant: str, rgb_backbone: str, event_backbone: str, model_path) -> str:
    model_path = Path(model_path).expanduser()
    parent = model_path.parent.name
    model_label = f"{parent}_{model_path.stem}" if parent not in ("", ".") else model_path.stem
    return f"{model_variant}_rgb_{rgb_backbone}_event_{event_backbone}_{model_label}"


def batched_nms_coordinate_trick(boxes, scores, labels, iou_threshold, width, height):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_dimension = max(width, height)
    offsets = labels * float(max_dimension + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)


def convert_to_evaluation_format(data, bbox_attr: str = "bbox"):
    targets = []
    for item in data.to_data_list():
        boxes = getattr(item, bbox_attr).clone()
        boxes[:, 2:4] += boxes[:, :2]
        targets.append({"boxes": boxes[:, :4], "labels": boxes[:, 4].long()})
    return targets


def postprocess_network_output(
    prediction,
    num_classes,
    conf_thre=0.01,
    nms_thre=0.65,
    height=640,
    width=640,
    filtering=True,
):
    prediction = prediction.clone()
    prediction[..., :2] -= prediction[..., 2:4] / 2
    prediction[..., 2:4] += prediction[..., :2]
    output = []
    for image_prediction in prediction:
        if len(image_prediction) == 0:
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=prediction.device),
                "scores": torch.zeros(0, dtype=torch.float32, device=prediction.device),
                "labels": torch.zeros(0, dtype=torch.long, device=prediction.device),
            })
            continue
        class_confidence, class_prediction = torch.max(
            image_prediction[:, 5:5 + num_classes], 1, keepdim=True
        )
        image_prediction[:, 4:5] *= class_confidence
        confidence_mask = image_prediction[:, 4] * class_confidence.squeeze() >= conf_thre
        detections = torch.cat((image_prediction[:, :5], class_prediction), 1)
        if filtering:
            detections = detections[confidence_mask]
        if len(detections) == 0:
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=prediction.device),
                "scores": torch.zeros(0, dtype=torch.float32, device=prediction.device),
                "labels": torch.zeros(0, dtype=torch.long, device=prediction.device),
            })
            continue
        keep_indices = batched_nms_coordinate_trick(
            detections[:, :4], detections[:, 4], detections[:, 5], nms_thre, width, height
        )
        if filtering:
            detections = detections[keep_indices]
        output.append({
            "boxes": detections[:, :4],
            "scores": detections[:, 4],
            "labels": detections[:, -1].long(),
        })
    return output
