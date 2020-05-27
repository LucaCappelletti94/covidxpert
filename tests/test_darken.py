from covidxpert import load_image
from covidxpert.utils import darken
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_darken():
    for path in tqdm(glob("test_images/*"), desc="Testing darken"):
        original = load_image(path)
        darkened_standard = darken(original)
        darkened_configured = darken(original, 4, (5, 5))
        darkened_configured_2 = darken(original, 1, (7, 7))

        fig, axes = plt.subplots(ncols=2, nrows=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(darkened_standard, cmap="gray")
        axes[1].set_title("Darkened image (standard parameters)")

        axes[2].imshow(darkened_configured, cmap="gray")
        axes[2].set_title("Darkened image (clip=4, Tuple=(5, 5))")

        axes[3].imshow(darkened_configured, cmap="gray")
        axes[3].set_title("Darkened image (clip=1, Tuple=(7, 7))")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"darken/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
