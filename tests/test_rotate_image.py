from covidxpert import blur_bbox, perspective_correction
from covidxpert.utils import rotate_image, load_image
from glob import glob
import matplotlib.pyplot as plt
import os

def test_rotate_image():
    path = glob("tests/test_images/*.jpg")[0]
    normalized = blur_bbox(perspective_correction(load_image(path)))
    rotated = rotate_image(normalized, 45)
    fig, axes = plt.subplots(ncols=2)
    axes = axes.ravel()

    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title("Original image")

    axes[1].imshow(rotated, cmap="gray")
    axes[1].set_title("Rotated")

    fig.tight_layout()
    path = f"tests/rotate_image/{os.path.basename(path)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

