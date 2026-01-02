# import os
# import json
# import numpy as np
# from PIL import Image, ImageDraw

# # -----------------------------
# # CONFIG
# # -----------------------------
# JSON_FOLDER = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/labels/post_disaster"       # folder containing xView2 JSONs
# OUTPUT_MASK_FOLDER = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/masks"
# IMAGE_SIZE = (1024, 1024)                  # width, height from metadata

# # Class mapping
# DAMAGE_MAP = {
#     "no-damage": 1,
#     "minor-damage": 2,
#     "major-damage": 3,
#     "destroyed": 4
# }

# # -----------------------------
# # HELPER: convert WKT → polygon list
# # -----------------------------
# def parse_wkt_polygon(wkt_str):
#     # Example: "POLYGON ((x y, x y, ...))"
#     wkt_str = wkt_str.replace("POLYGON ((", "").replace("))", "")
#     pts = []
#     for pair in wkt_str.split(","):
#         x, y = pair.strip().split()
#         pts.append((float(x), float(y)))
#     return pts

# # -----------------------------
# # MAIN: Convert one JSON → mask
# # -----------------------------
# def generate_mask(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     mask = Image.new("L", IMAGE_SIZE, 0)  # grayscale 0–4 classes
#     drawer = ImageDraw.Draw(mask)

#     if "xy" not in data["features"]:
#         print(f"Skipping {json_path}: No XY polygons.")
#         return

#     for obj in data["features"]["xy"]:
#         subtype = obj["properties"]["subtype"].strip()
#         wkt = obj["wkt"]

#         if subtype not in DAMAGE_MAP:
#             continue

#         class_id = DAMAGE_MAP[subtype]
#         polygon = parse_wkt_polygon(wkt)

#         drawer.polygon(polygon, fill=class_id)

#     # Save mask
#     basename = os.path.basename(json_path).replace(".json", "_mask.png")
#     save_path = os.path.join(OUTPUT_MASK_FOLDER, basename)
#     mask.save(save_path)
#     print("Generated mask:", save_path)

# # -----------------------------
# # RUN BATCH PROCESS
# # -----------------------------
# def main():
#     os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

#     json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

#     print("\n=== Generating masks for entire folder ===\n")
#     for jf in json_files:
#         generate_mask(os.path.join(JSON_FOLDER, jf))

#     print("\nDONE! All masks generated.\n")

# if __name__ == "__main__":
#     main()



import os
from PIL import Image

# ---------------------
# CONFIG
# ---------------------
MASK_DIR = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/masks"           # your existing mask folder
OUT_DIR = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/masks_512"             # output folder
PATCH_SIZE = 512

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------
# FUNCTION
# ---------------------
def split_mask(mask_path):
    base = os.path.splitext(os.path.basename(mask_path))[0]
    mask = Image.open(mask_path)

    w, h = mask.size
    patch_id = 0

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):

            patch = mask.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            patch.save(f"{OUT_DIR}/{base}_patch{patch_id}.png")
            patch_id += 1

    print(f"✔ {base} → {patch_id} patches")


# ---------------------
# MAIN
# ---------------------
if __name__ == "__main__":
    files = [f for f in os.listdir(MASK_DIR) if f.endswith((".png", ".jpg"))]

    for f in files:
        split_mask(os.path.join(MASK_DIR, f))
