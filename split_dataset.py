import os
import shutil
import random

# -------------------------
# Config
# -------------------------
DATA_DIR = "/content/data"  # change if needed
VAL_RATIO = 0.2             # 20% validation

pre_dir = os.path.join(DATA_DIR, "pre")
post_dir = os.path.join(DATA_DIR, "post")
mask_dir = os.path.join(DATA_DIR, "masks")

# New destination roots
train_root = os.path.join(DATA_DIR, "train")
val_root = os.path.join(DATA_DIR, "val")

for root in [train_root, val_root]:
    for sub in ["pre", "post", "masks"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)

# -------------------------
# Gather files
# -------------------------
pre_files = sorted([
    f for f in os.listdir(pre_dir) 
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# Shuffle for randomness
random.shuffle(pre_files)

# Compute split index
split_idx = int(len(pre_files) * (1 - VAL_RATIO))

train_files = pre_files[:split_idx]
val_files = pre_files[split_idx:]

# -------------------------
# Helper to move matching files
# -------------------------
def move_set(files, destination_root):
    for pre_file in files:

        # Find corresponding post & mask filenames
        post_file = pre_file.replace("pre_disaster", "post_disaster").replace("pre_", "post_")
        mask_file = post_file.replace("_post_disaster_patch", "_post_disaster_mask_patch")


        pre_src = os.path.join(pre_dir, pre_file)
        post_src = os.path.join(post_dir, post_file)
        mask_src = os.path.join(mask_dir, mask_file)

        # Skip if any file is missing
        if not (os.path.exists(post_src) and os.path.exists(mask_src)):
            continue

        shutil.copy(pre_src, os.path.join(destination_root, "pre", pre_file))
        shutil.copy(post_src, os.path.join(destination_root, "post", post_file))
        shutil.copy(mask_src, os.path.join(destination_root, "masks", mask_file))

    print(f"Moved {len(files)} items → {destination_root}")

# -------------------------
# Move files
# -------------------------
move_set(train_files, train_root)
move_set(val_files, val_root)

print("Dataset split complete.")
