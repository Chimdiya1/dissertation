import csv
import json
from pathlib import Path
import numpy as np

# ----------------------------
# EDIT PATHS
# ----------------------------
POST_JSON_FOLDER = Path("/content/train/labels/post_disaster")
PRED_BUILDINGS_CSV = Path("/content/train/building_preds/building_predictions_all.csv")

# Optional: evaluate only a split (scene_id list file). Leave as None to eval all scenes.
SPLIT_SCENE_IDS_TXT = "/content/train/splits/val_ids.txt"  # e.g. Path("/content/train/splits/val_ids.txt")

# ----------------------------
# Label maps
# ----------------------------
LABEL_TO_ID = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

CLASSES = [1, 2, 3, 4]  # evaluate damage classes only (ignore background=0)


def load_split_scene_ids(path: Path):
    if path is None:
        return None
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return set(ids)


def scene_id_from_post_json_name(name: str) -> str:
    return Path(name).name.replace("_post_disaster.json", "")


def load_truth_from_jsons(json_folder: Path, allowed_scenes=None):
    """
    Returns dict: (scene_id, uid) -> true_label_id (1..4)
    """
    truth = {}
    for p in sorted(json_folder.glob("*_post_disaster.json")):
        scene_id = scene_id_from_post_json_name(p.name)
        if allowed_scenes is not None and scene_id not in allowed_scenes:
            continue

        data = json.loads(p.read_text())
        feats = data.get("features", {}).get("xy", [])
        for feat in feats:
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            uid = props.get("uid")
            subtype = props.get("subtype")
            if not uid or not subtype:
                continue
            if subtype not in LABEL_TO_ID:
                continue  # skip anything unexpected
            truth[(scene_id, uid)] = LABEL_TO_ID[subtype]
    return truth


def load_preds_from_csv(pred_csv: Path, allowed_scenes=None):
    """
    Returns dict: (scene_id, uid) -> pred_label_id (0..4)
    """
    preds = {}
    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            scene_id = r["scene_id"].strip()
            if allowed_scenes is not None and scene_id not in allowed_scenes:
                continue
            uid = r["uid"].strip()
            pred_id = int(r["pred_label_id"])
            preds[(scene_id, uid)] = pred_id
    return preds


def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def precision_recall_f1_from_cm(cm):
    # rows=true, cols=pred
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = cm.sum(axis=1).astype(np.int64)
    return precision, recall, f1, support


def main():
    allowed_scenes = load_split_scene_ids(SPLIT_SCENE_IDS_TXT) if SPLIT_SCENE_IDS_TXT else None

    truth = load_truth_from_jsons(POST_JSON_FOLDER, allowed_scenes)
    preds = load_preds_from_csv(PRED_BUILDINGS_CSV, allowed_scenes)

    # Join on (scene_id, uid)
    y_true, y_pred = [], []
    missing_pred = 0
    for k, t in truth.items():
        if k not in preds:
            missing_pred += 1
            continue
        p = preds[k]

        # evaluate only 1..4 (ignore background pred=0 by skipping)
        if p not in CLASSES:
            continue

        y_true.append(t)
        y_pred.append(p)

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    print("Matched buildings used:", len(y_true))
    print("Truth buildings missing prediction:", missing_pred)

    if len(y_true) == 0:
        print("No matched buildings to evaluate. Check paths and filenames.")
        return

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, CLASSES)
    prec, rec, f1, sup = precision_recall_f1_from_cm(cm)

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * sup) / max(np.sum(sup), 1))

    print("\nOverall")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")

    print("\nPer-class")
    for i, c in enumerate(CLASSES):
        print(f"{ID_TO_LABEL[c]:12s} | P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} | support={sup[i]}")

    print("\nConfusion matrix (rows=true, cols=pred) for classes [1..4]")
    print("Order:", [ID_TO_LABEL[c] for c in CLASSES])
    print(cm)


if __name__ == "__main__":
    main()
