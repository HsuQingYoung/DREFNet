# DREFNet

## Model Weights

Download the ONNX model from [pretrained model](https://1drv.ms/f/c/961aa4cfbaf1dfe1/IgBnQy2vI0SvQqB42Lt78s7eAS9ldB5hlKK4FUWgnvM_-SY?e=tIgbdw) and place it at `weights/best_model_map.onnx`.

## Environment

Download DSEC from the [official DSEC website](https://dsec.ifi.uzh.ch/) and the detection annotations from the [DSEC-Detection page](https://dsec.ifi.uzh.ch/dsec-detection/). Download or prepare the three ProVe-filtered LaMa RGB variants, then set the paths below once:

```bash
DREFNET_DIR=/path/to/work/DREFNet

DSEC_ROOT=/path/to/dsec
EMERGE_1_ROOT=/path/to/dsec-emerge-1/dsec
EMERGE_2_ROOT=/path/to/dsec-emerge-2/dsec
EMERGE_3_ROOT=/path/to/dsec-emerge-3/dsec

cd "$DREFNET_DIR"
conda env create -f environment.yml
conda activate drefnet
```

## Dataset Structure

The DSEC root contains `train/` and `test/`; each evaluation sequence needs the following files:

```text
$DSEC_ROOT/
├── train/<sequence>/
│   ├── images/
│   │   ├── timestamps.txt
│   │   └── left/distorted/*.png
│   ├── events/left/events_2x.h5
│   └── object_detections/left/tracks.npy
└── test/<sequence>/                 # same structure
```

Each LaMa inpainting variant provides the corresponding RGB frames and removed-target annotations; events and detection tracks continue to come from `$DSEC_ROOT`:

```text
$EMERGE_ROOT/dsec/
├── train/<sequence>/images/left/
│   ├── distorted/*.png
│   └── distorted_removed_targets.json
└── test/<sequence>/images/left/     # same structure
    ├── distorted/*.png
    └── distorted_removed_targets.json
```

## Data Processing

Use [LaMa](https://github.com/advimman/lama) to remove targets from the DSEC RGB frames and generate three inpainting levels. The resulting RGB datasets are the three `EMERGE_*_ROOT` datasets. The bundled `annotations/removed_targets/` files record which targets were removed in each frame; copy them into the corresponding RGB dataset roots before evaluation:

```bash
copy_removed_targets() {
  local source_root="$1"
  local rgb_root="$2"
  if [[ "$(basename "$rgb_root")" != dsec ]]; then
    rgb_root="$rgb_root/dsec"
  fi
  mkdir -p "$rgb_root"
  cp -a "$source_root"/. "$rgb_root"/
}

copy_removed_targets annotations/removed_targets/dsec-emerge-1/dsec "$EMERGE_1_ROOT"
copy_removed_targets annotations/removed_targets/dsec-emerge-2/dsec "$EMERGE_2_ROOT"
copy_removed_targets annotations/removed_targets/dsec-emerge-3/dsec "$EMERGE_3_ROOT"
```

Generate an exclusion file using the paths you set above. It remaps the bundled `dsec-emerge-1/2/3` entries to the actual directory names of your three RGB roots:

```bash
python scripts/prepare_exclude_json.py \
  --exclude_json annotations/exclude_rc_s.json \
  --alt_rgb_dataset_directory "$EMERGE_1_ROOT" "$EMERGE_2_ROOT" "$EMERGE_3_ROOT" \
  --output_json outputs/exclude_rc_s.local.json
```

Use `outputs/exclude_rc_s.local.json` with `--exclude_json` in either evaluation command.

## 推理命令

### Emergence recall

```bash
python scripts/eval_emergence_recall.py \
  --onnx weights/best_model_map.onnx \
  --dataset_directory "$DSEC_ROOT" \
  --alt_rgb_dataset_directory "$EMERGE_1_ROOT" "$EMERGE_2_ROOT" "$EMERGE_3_ROOT" \
  --exclude_json outputs/exclude_rc_s.local.json \
  --only_perfect_tracks 0 --shared_track_interpolation 1 \
  --provider CUDAExecutionProvider \
  --output_directory outputs/emergence_recall
```

### Inter-frame mAP

```bash
python scripts/eval_map_interframe_all.py \
  --onnx weights/best_model_map.onnx \
  --dataset_directory "$DSEC_ROOT" \
  --alt_rgb_dataset_directory "$EMERGE_1_ROOT" "$EMERGE_2_ROOT" "$EMERGE_3_ROOT" \
  --exclude_json outputs/exclude_rc_s.local.json \
  --split test --only_perfect_tracks 0 --shared_track_interpolation 1 \
  --provider CUDAExecutionProvider --batch_size 1 \
  --output_directory outputs/interframe_map
```
