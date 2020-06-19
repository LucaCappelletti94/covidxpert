import os
from glob import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from covidxpert import load_image, perspective_correction
from covidxpert.utils import automated_msrcr


def test_retinex():
    for path in tqdm(glob("tests/test_images/*"), desc="Test pipeline"):
        original = perspective_correction(load_image(path))
        retinex_image = automated_msrcr(original, [15, 80, 250])

        fig, axes = plt.subplots(ncols=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(retinex_image, cmap="gray")
        axes[1].set_title("Retinex image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/retinex/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
