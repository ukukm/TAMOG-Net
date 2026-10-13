import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tamog_net import TAMOGNet

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "tea": {
        "root": ROOT / "samples/tea",
        "weight": ROOT / "weights/tamog_net_tea_seed3407.pth",
        "model_index": {
            "Basilepta fulvipes": 0,
            "Tea anthracnose": 1,
            "Heat stress": 2,
            "Healthy": 3,
        },
    },
    "rice": {
        "root": ROOT / "samples/rice",
        "weight": ROOT / "weights/tamog_net_rice_seed3407.pth",
        "model_index": {
            "Rice bacterial blight": 0,
            "Rice leaf roller": 1,
            "Rice stem borer": 2,
            "Healthy": 3,
            "Rice leaf blast": 4,
        },
    },
}


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def evaluate(name, device, output_root):
    cfg = CONFIG[name]
    dataset = datasets.ImageFolder(cfg["root"], transform=make_transform())
    if set(dataset.classes) != set(cfg["model_index"]):
        raise ValueError(f"Unexpected {name} labels: {dataset.classes}")
    output_to_folder = [cfg["model_index"][label] for label in dataset.classes]
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model = TAMOGNet(
        num_classes=len(dataset.classes), num_blocks=5, width_mult=1.25,
        expansion=2.0, aux_loss=False, variant="tamog",
    )
    state = torch.load(cfg["weight"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    rows = []
    offset = 0
    for images, targets in loader:
        model_probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
        probabilities = model_probabilities[:, output_to_folder]
        predictions = probabilities.argmax(1)
        for i, target in enumerate(targets.tolist()):
            predicted = predictions[i].item()
            rows.append({
                "image": Path(dataset.samples[offset + i][0]).relative_to(cfg["root"]).as_posix(),
                "true_label": dataset.classes[target],
                "predicted_label": dataset.classes[predicted],
                "confidence": f"{probabilities[i, predicted].item():.8f}",
                "correct": str(target == predicted),
            })
        offset += len(targets)

    correct = [row for row in rows if row["correct"] == "True"]
    incorrect = [row for row in rows if row["correct"] == "False"]
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image", "true_label", "predicted_label", "confidence", "correct"]
    for filename, records in (("all_predictions.csv", rows), ("correct.csv", correct),
                              ("incorrect.csv", incorrect)):
        with (out_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "dataset": name,
        "seed": 3407,
        "weight": cfg["weight"].name,
        "num_images": len(rows),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": len(correct) / len(rows),
        "labels_by_model_output_index": [
            label for label, _ in sorted(cfg["model_index"].items(), key=lambda item: item[1])
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"{name}: {len(correct)}/{len(rows)} correct ({summary['accuracy']:.2%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", default=str(ROOT / "test_results"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    torch.set_num_threads(4)
    for name in ("tea", "rice"):
        evaluate(name, args.device, Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
