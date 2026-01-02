import os
from pathlib import Path

def check_file_correspondence():
    """
    Check if each post_disaster image has corresponding:
    - pre_disaster image
    - post_disaster label
    - pre_disaster label
    """
    base_dir = Path(__file__).parent
    
    # Define directories
    post_img_dir = base_dir / "train/images/post_disaster"
    pre_img_dir = base_dir / "train/images/pre_disaster"
    post_label_dir = base_dir / "train/labels/post_disaster"
    pre_label_dir = base_dir / "train/labels/pre_disaster"
    
    # Check if directories exist
    if not post_img_dir.exists():
        print(f"Error: {post_img_dir} does not exist")
        return
    
    # Get all post-disaster images
    post_images = sorted([f for f in os.listdir(post_img_dir) if not f.startswith('.')])
    
    print(f"Found {len(post_images)} post-disaster images\n")
    
    missing_pre_images = []
    missing_post_labels = []
    missing_pre_labels = []
    complete_sets = []
    
    for img_file in post_images:
        # Get base name (assuming format: name_post_disaster.ext)
        base_name = img_file.replace('_post_disaster', '_pre_disaster')
        label_name_post = img_file.replace('.png', '.json').replace('.jpg', '.json')
        label_name_pre = base_name.replace('.png', '.json').replace('.jpg', '.json')
        
        # Check pre-disaster image
        pre_img_path = pre_img_dir / base_name
        has_pre_img = pre_img_path.exists()
        
        # Check post-disaster label
        post_label_path = post_label_dir / label_name_post
        has_post_label = post_label_path.exists()
        
        # Check pre-disaster label
        pre_label_path = pre_label_dir / label_name_pre
        has_pre_label = pre_label_path.exists()
        
        # Track missing files
        if not has_pre_img:
            missing_pre_images.append(img_file)
        if not has_post_label:
            missing_post_labels.append(img_file)
        if not has_pre_label:
            missing_pre_labels.append(img_file)
        
        # Track complete sets
        if has_pre_img and has_post_label and has_pre_label:
            complete_sets.append(img_file)
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total post-disaster images: {len(post_images)}")
    print(f"Complete sets (all 4 files present): {len(complete_sets)}")
    print(f"\nMissing pre-disaster images: {len(missing_pre_images)}")
    print(f"Missing post-disaster labels: {len(missing_post_labels)}")
    print(f"Missing pre-disaster labels: {len(missing_pre_labels)}")
    
    # Print details
    if missing_pre_images:
        print("\n" + "=" * 80)
        print("MISSING PRE-DISASTER IMAGES")
        print("=" * 80)
        for f in missing_pre_images[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_pre_images) > 10:
            print(f"  ... and {len(missing_pre_images) - 10} more")
    
    if missing_post_labels:
        print("\n" + "=" * 80)
        print("MISSING POST-DISASTER LABELS")
        print("=" * 80)
        for f in missing_post_labels[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_post_labels) > 10:
            print(f"  ... and {len(missing_post_labels) - 10} more")
    
    if missing_pre_labels:
        print("\n" + "=" * 80)
        print("MISSING PRE-DISASTER LABELS")
        print("=" * 80)
        for f in missing_pre_labels[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_pre_labels) > 10:
            print(f"  ... and {len(missing_pre_labels) - 10} more")
    
    if len(complete_sets) == len(post_images):
        print("\n" + "=" * 80)
        print("✓ All post-disaster images have complete corresponding files!")
        print("=" * 80)
    
    # Save detailed report to file
    report_path = base_dir / "correspondence_report.txt"
    with open(report_path, 'w') as f:
        f.write("FILE CORRESPONDENCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total post-disaster images: {len(post_images)}\n")
        f.write(f"Complete sets: {len(complete_sets)}\n\n")
        
        f.write(f"Missing pre-disaster images: {len(missing_pre_images)}\n")
        for img in missing_pre_images:
            f.write(f"  - {img}\n")
        
        f.write(f"\nMissing post-disaster labels: {len(missing_post_labels)}\n")
        for img in missing_post_labels:
            f.write(f"  - {img}\n")
        
        f.write(f"\nMissing pre-disaster labels: {len(missing_pre_labels)}\n")
        for img in missing_pre_labels:
            f.write(f"  - {img}\n")
        
        f.write(f"\nComplete sets ({len(complete_sets)} files):\n")
        for img in complete_sets:
            f.write(f"  - {img}\n")
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    check_file_correspondence()
