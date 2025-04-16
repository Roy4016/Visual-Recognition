import os
import json
import torch
import torchvision
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import torchvision.transforms as T
import pandas as pd
#---------------------------
# Valid
#---------------------------
def evaluate(model, dataloader):
    total_loss = 0
    model.train()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = list(img.to("cuda") for img in imgs)
            targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -----------------------------
# Dataset + Loader
# -----------------------------
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/train", annotation_path="data/train.json", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(annotation_path, "r") as file:
            coco = json.load(file)

        self.image_map = {img["id"]: img["file_name"] for img in coco["images"]}
        self.annotations_map = {img_id: [] for img_id in self.image_map}
        for ann in coco["annotations"]:
            self.annotations_map[ann["image_id"]].append(ann)

        self.ids = list(self.image_map.keys())
        self.id_to_classname = {cat["id"]: cat["name"] for cat in coco["categories"]}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root_dir, self.image_map[img_id])
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations_map[img_id]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in (ann["bbox"] for ann in anns)]
        labels = [ann["category_id"] for ann in anns]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir="data/test", transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(test_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        img_path = os.path.join(self.test_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, os.path.splitext(file_name)[0]

def prepare_dataloader(root, annotation, transform, batch_size, workers, shuffle=True, ratio=-1):
    dataset = TrainDataset(root_dir=root, annotation_path=annotation, transform=transform)
    if 0 < ratio < 1:
        subset_len = int(len(dataset) * ratio)
        dataset = Subset(dataset, range(subset_len))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    return loader, dataset

# -----------------------------
# Generate Prediction CSV
# -----------------------------
def generate_pred_csv(pred_json_path, output_csv_path, id_to_classname):
    with open(pred_json_path) as f:
        preds = json.load(f)
    image_preds = {}
    for pred in preds:
        image_id, bbox, label, score = pred['image_id'], pred['bbox'], pred['category_id'], pred['score']
        image_preds.setdefault(image_id, []).append((bbox[0], label, score))
    rows = []
    for image_id, preds in image_preds.items():
        preds_sorted = sorted(preds, key=lambda x: x[0])
        pred_label = ''.join([id_to_classname[label] for _, label, _ in preds_sorted])
        rows.append({"image_id": image_id, "pred_label": pred_label})
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

# -----------------------------
# Inference
# -----------------------------
def run_inference():
    test_dir = "data/test"
    best_model_path = "best_model.pth"
    pred_json_path = "pred.json"
    pred_csv_path = "pred.csv"

    transform = T.ToTensor()
    id_to_classname = TrainDataset(
        root_dir="data/train", 
        annotation_path="data/train.json"
    ).id_to_classname

    num_classes = len(id_to_classname)

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.load_state_dict(torch.load(best_model_path))
    model.eval().cuda()

    test_ds = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for imgs, image_ids in tqdm(test_loader):
            imgs = list(img.cuda() for img in imgs)
            outputs = model(imgs)
            for image_id, output in zip(image_ids, outputs):
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if score < 0.3:
                        continue
                    x_min, y_min, x_max, y_max = box
                    w, h = x_max - x_min, y_max - y_min
                    results.append({
                        "image_id": int(image_id),
                        "bbox": [float(x_min), float(y_min), float(w), float(h)],
                        "score": float(score),
                        "category_id": int(label)
                    })

    with open(pred_json_path, 'w') as f:
        json.dump(results, f)
    generate_pred_csv(pred_json_path, pred_csv_path, id_to_classname)

    df = pd.read_csv(pred_csv_path)
    pred_dict = dict(zip(df["image_id"], df["pred_label"]))
    max_id = 10638
    full_ids = list(range(1, max_id + 1))
    rows = [{"image_id": i, "pred_label": pred_dict.get(i, -1)} for i in full_ids]
    lex_df = pd.DataFrame(rows)
    lex_df["image_id_str"] = lex_df["image_id"].astype(str)
    lex_df = lex_df.sort_values("image_id_str").drop(columns="image_id_str")
    lex_df.to_csv(pred_csv_path, index=False)

# -----------------------------
# Training
# -----------------------------
def train_model():
    train_dir = "data/train"
    val_dir = "data/valid"
    train_json = "data/train.json"
    val_json = "data/valid.json"
    best_model_path = "best_model.pth"

    transform = T.ToTensor()
    batch_size = 4
    num_workers = 4
    epochs = 5

    train_loader, train_dataset = prepare_dataloader(train_dir, train_json, transform, batch_size, num_workers, True)
    val_loader, _ = prepare_dataloader(val_dir, val_json, transform, batch_size, num_workers, False)

    id_to_classname = train_dataset.id_to_classname
    num_classes = len(id_to_classname)

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.to("cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"[Epoch {epoch+1}/{epochs}]")
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader):
            imgs = list(img.to("cuda") for img in imgs)
            targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("[!] Best model saved!")
# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "infer":
        run_inference()
