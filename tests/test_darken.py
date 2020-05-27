from covidxpert import load_image
from covidxpert.utils import darken
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_darken():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing darken"):
        original = load_image(path)
        darkened = darken(original)

        fig, axes = plt.subplots(ncols=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(darkened, cmap="gray")
        axes[1].set_title("Darkened image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/darken/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
