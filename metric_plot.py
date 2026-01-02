import matplotlib.pyplot as plt
import json
import os

# ------------------------------------------------------
# Load metrics.json — created during training or eval
# ------------------------------------------------------
def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data["epochs"], data["iou"], data["dice"]


# ------------------------------------------------------
# Plot IoU + Dice curves
# ------------------------------------------------------
def plot_metrics(epochs, iou, dice, save_path="metric_plot.png"):
    plt.figure(figsize=(8,5))

    plt.plot(epochs, iou, marker="o", label="IoU")
    plt.plot(epochs, dice, marker="s", label="Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Segmentation Performance Across Epochs")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path}")

    plt.show()


# ------------------------------------------------------
# Main Entry
# ------------------------------------------------------
def main():
    metrics_path = "/content/checkpoints/metrics.json"   # <-- update if needed
    save_path = "/content/metric_plot.png"

    epochs, iou, dice = load_metrics(metrics_path)
    plot_metrics(epochs, iou, dice, save_path)


if __name__ == "__main__":
    main()
