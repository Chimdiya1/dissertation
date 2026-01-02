import os
from PIL import Image

# -----------------------------
# Config
# -----------------------------
pre_dir = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/pre_disaster"       # folder with pre-disaster images
post_dir = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/post_disaster"     # folder with post-disaster images
output_pre_dir = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/pre_disaster_patch"
output_post_dir = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/post_disaster_patch"
patch_size = 512

# Create output folders if they don't exist
os.makedirs(output_pre_dir, exist_ok=True)
os.makedirs(output_post_dir, exist_ok=True)

# -----------------------------
# List images
# -----------------------------
pre_images = sorted([f for f in os.listdir(pre_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
post_images = sorted([f for f in os.listdir(post_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
# -----------------------------
# Generate patches
# -----------------------------
for pre_img_name, post_img_name in zip(pre_images, post_images):
    pre_img_path = os.path.join(pre_dir, pre_img_name)
    post_img_path = os.path.join(post_dir, post_img_name)

    pre_img = Image.open(pre_img_path)
    post_img = Image.open(post_img_path)

    # Ensure images are the same size
    if pre_img.size != post_img.size:
        print(f"Skipping {pre_img_name} / {post_img_name}: sizes differ")
        continue

    width, height = pre_img.size
    patch_id = 0

    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            box = (left, top, min(left + patch_size, width), min(top + patch_size, height))

            pre_patch = pre_img.crop(box)
            post_patch = post_img.crop(box)

            # Save patches
            pre_patch.save(os.path.join(output_pre_dir, f"{os.path.splitext(pre_img_name)[0]}_patch{patch_id}.png"))
            post_patch.save(os.path.join(output_post_dir, f"{os.path.splitext(post_img_name)[0]}_patch{patch_id}.png"))

            patch_id += 1

    print(f"Generated {patch_id} patches for {pre_img_name} / {post_img_name}")

print("Patch generation complete.")
