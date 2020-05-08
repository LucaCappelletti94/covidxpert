from covidxpert import load_image, perspective_correction
from covidxpert.utils.remove_artefacts import remove_artefacts
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from glob import glob


def test_remove_artifacts():
    for path in tqdm(glob("tests/test_images/*")):
        original = load_image(path)
        cut_image = perspective_correction(original)

        fig, axes = plt.subplots(ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(cut_image, cmap="gray")
        axes[1].set_title("Perspective correction")

        cleared_image = remove_artefacts(cut_image)

        axes[2].imshow(cleared_image, cmap="gray")
        axes[2].set_title("Removed artifacts")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/remove_artifacts/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
