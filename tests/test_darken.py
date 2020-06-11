from covidxpert import load_image, perspective_correction
from covidxpert import blur_bbox
from covidxpert.utils import darken
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_darken():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing darken"):
        original = load_image(path)
        cut_image = perspective_correction(original)
        blurred_image = blur_bbox(cut_image)

        darkened_standard = darken(blurred_image)

        fig, axes = plt.subplots(ncols=2)
        axes = axes.ravel()

        axes[0].imshow(blurred_image, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(darkened_standard, cmap="gray")
        axes[1].set_title("Darkened (defaults)")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/darken/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
