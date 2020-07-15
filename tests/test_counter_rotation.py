import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from covidxpert import perspective_correction, blur_bbox, counter_rotate
from covidxpert.utils import load_image, remove_artefacts


def test_perspective_correction():
    for path in tqdm(glob("tests/test_images/*") + glob("tests/rotated_images/*"), desc="Test counter rotation"):
        original = remove_artefacts(blur_bbox(perspective_correction(load_image(path))))
        rotated, _, x = counter_rotate(original)

        fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
        axes = axes.ravel()

        axes[0].set_title("Original image")
        axes[0].imshow(original, cmap="gray")

        axes[1].set_title("Rotated image")
        axes[1].imshow(rotated, cmap="gray")
        axes[1].axvline(x)
        fig.tight_layout()
        path = f"tests/counter_rotation/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
