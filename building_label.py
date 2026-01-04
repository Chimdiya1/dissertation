import os
import json
import re
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ----------------------------
# Paths (EDIT)
# ----------------------------
POST_JSON_FOLDER = Path("/content/train/labels/post_disaster")   # folder of post_disaster JSONs
PRED_MASK_FOLDER = Path("/content/train/preds_fullscene")        # folder of *_pred_mask.png from Step 11
OUT_FOLDER = Path("/content/train/building_preds")
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Label maps
# ----------------------------
ID_TO_LABEL = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

# Decision rule: "majority" or "max"
DECISION_RULE = "majority"


def parse_wkt_polygon(wkt: str):
    """Parse POLYGON WKT -> list of rings (each ring list[(x,y)])."""
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        raise ValueError("Expected POLYGON WKT")

    inner = wkt[wkt.find("("):].strip()   # "((...))"
    inner = inner[2:-2].strip()           # remove outer "((" and "))"
    ring_strs = re.split(r"\)\s*,\s*\(", inner)

    rings = []
    for ring_str in ring_strs:
        pts = []
        for pair in ring_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            x_str, y_str = pair.split()
            pts.append((float(x_str), float(y_str)))
        if len(pts) >= 3:
            rings.append(pts)

    if not rings:
        raise ValueError("Failed to parse polygon")
    return rings


def clamp_points(points, width, height):
    out = []
    for x, y in points:
        x = 0.0 if x < 0 else (width - 1.0 if x > width - 1 else x)
        y = 0.0 if y < 0 else (height - 1.0 if y > height - 1 else y)
        out.append((x, y))
    return out


def scene_id_from_post_json_name(json_name: str) -> str:
    # hurricane-harvey_00000000_post_disaster.json -> hurricane-harvey_00000000
    name = Path(json_name).name
    return name.replace("_post_disaster.json", "")


def pred_mask_name(scene_id: str) -> str:
    # hurricane-harvey_00000000 -> hurricane-harvey_00000000_pred_mask.png
    return f"{scene_id}_pred_mask.png"


def label_building_from_pixels(pixels: np.ndarray):
    """
    pixels: 1D array of class ids inside polygon
    Returns: (label_id, confidence, counts)
    """
    if pixels.size == 0:
        return 0, 0.0, None

    if DECISION_RULE == "max":
        # ignore background unless everything is background
        non_bg = pixels[pixels != 0]
        label_id = int(non_bg.max()) if non_bg.size else 0
        conf = 1.0  # not meaningful for max rule
        return label_id, conf, None

    # majority vote (ignore background)
    non_bg = pixels[pixels != 0]
    if non_bg.size == 0:
        return 0, 0.0, None

    counts = np.bincount(non_bg, minlength=5)  # indices 0..4
    label_id = int(np.argmax(counts))
    conf = float(counts[label_id] / max(non_bg.size, 1))
    return label_id, conf, counts


def process_scene(post_json_path: Path, pred_mask_path: Path, out_csv_path: Path):
    data = json.loads(post_json_path.read_text())

    meta = data.get("metadata", {})
    width = int(meta.get("width", meta.get("original_width", 1024)))
    height = int(meta.get("height", meta.get("original_height", 1024)))

    pred_mask = np.array(Image.open(pred_mask_path), dtype=np.uint8)
    if pred_mask.shape != (height, width):
        raise ValueError(f"Pred mask shape {pred_mask.shape} != ({height},{width})")

    feats = data.get("features", {}).get("xy", [])

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "uid", "pred_label_id", "pred_label", "confidence", "pixel_count"])

        for feat in feats:
            props = feat.get("properties", {})
            if props.get("feature_type") != "building":
                continue

            uid = props.get("uid", "")
            wkt = feat.get("wkt", "")
            if not uid or not wkt:
                continue

            rings = parse_wkt_polygon(wkt)
            outer = clamp_points(rings[0], width, height)

            # Rasterize polygon to a binary mask (only for this building)
            poly_img = Image.new("1", (width, height), 0)
            draw = ImageDraw.Draw(poly_img)
            draw.polygon(outer, fill=1)

            # If holes exist, clear them
            for hole_ring in rings[1:]:
                hole = clamp_points(hole_ring, width, height)
                draw.polygon(hole, fill=0)

            poly_mask = np.array(poly_img, dtype=bool)
            pixels = pred_mask[poly_mask]

            label_id, conf, _ = label_building_from_pixels(pixels)
            writer.writerow([scene_id_from_post_json_name(post_json_path.name), uid, label_id, ID_TO_LABEL[label_id], f"{conf:.4f}", int(pixels.size)])


def main():
    json_files = sorted(POST_JSON_FOLDER.glob("*_post_disaster.json"))
    if not json_files:
        print("No post_disaster JSONs found in:", POST_JSON_FOLDER)
        return

    combined_path = OUT_FOLDER / "building_predictions_all.csv"
    with open(combined_path, "w", newline="") as f_all:
        w_all = csv.writer(f_all)
        w_all.writerow(["scene_id", "uid", "pred_label_id", "pred_label", "confidence", "pixel_count"])

        ok, fail = 0, 0

        for post_json_path in json_files:
            scene_id = scene_id_from_post_json_name(post_json_path.name)
            pm_path = PRED_MASK_FOLDER / pred_mask_name(scene_id)

            if not pm_path.exists():
                print("[SKIP] missing pred mask for", scene_id)
                continue

            out_csv = OUT_FOLDER / f"{scene_id}_building_preds.csv"

            try:
                process_scene(post_json_path, pm_path, out_csv)

                # append to combined
                with open(out_csv, "r", newline="") as f_scene:
                    r = csv.reader(f_scene)
                    next(r, None)  # skip header
                    for row in r:
                        w_all.writerow(row)

                ok += 1
                print("[OK]", scene_id, "->", out_csv)

            except Exception as e:
                fail += 1
                print("[FAIL]", scene_id, ":", e)

    print("\nDone.")
    print("Combined CSV:", combined_path)


if __name__ == "__main__":
    main()
