"""Rewrite bundled RGB-exclusion keys for the supplied dataset roots."""

import argparse
import json
from pathlib import Path


SOURCE_DATASET_NAMES = (
    "dsec-emerge-1",
    "dsec-emerge-2",
    "dsec-emerge-3",
)


def dataset_name_for_root(root: Path) -> str:
    root = root.expanduser()
    return root.parent.name if root.name == "dsec" else root.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude_json", type=Path, required=True)
    parser.add_argument("--alt_rgb_dataset_directory", type=Path, nargs=3, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.exclude_json.expanduser()
    output_path = args.output_json.expanduser()
    if source_path.resolve() == output_path.resolve():
        parser.error("--output_json must differ from --exclude_json")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload)
    if not isinstance(datasets, dict):
        parser.error("exclude JSON must contain a 'datasets' object")

    missing_names = [name for name in SOURCE_DATASET_NAMES if name not in datasets]
    if missing_names:
        parser.error(f"exclude JSON is missing dataset keys: {', '.join(missing_names)}")

    target_names = [dataset_name_for_root(root) for root in args.alt_rgb_dataset_directory]
    if len(set(target_names)) != len(target_names):
        parser.error("the three RGB roots must resolve to distinct dataset names")

    name_mapping = dict(zip(SOURCE_DATASET_NAMES, target_names))
    remapped_datasets = {}
    for source_name, entries in datasets.items():
        target_name = name_mapping.get(source_name, source_name)
        if target_name in remapped_datasets:
            parser.error(f"dataset key collision after remapping: {target_name}")
        remapped_datasets[target_name] = entries

    if "datasets" in payload:
        payload["datasets"] = remapped_datasets
    else:
        payload = remapped_datasets

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    for source_name, target_name in name_mapping.items():
        print(f"{source_name} -> {target_name}")


if __name__ == "__main__":
    main()
