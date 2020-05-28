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
        darkened_configured = darken(blurred_image, 4, (5, 5))
        darkened_configured_2 = darken(blurred_image, 1, (7, 7))

        fig, axes = plt.subplots(ncols=2, nrows=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(darkened_standard, cmap="gray")
        axes[1].set_title("Darkened (defaults)")

        axes[2].imshow(darkened_configured, cmap="gray")
        axes[2].set_title("Darkened (clip=4, kernel=(5, 5))")

        axes[3].imshow(darkened_configured_2, cmap="gray")
        axes[3].set_title("Darkened (clip=1, kernel=(7, 7))")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/darken/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
