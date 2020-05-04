from covidxpert import load_image, perspective_correction
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from glob import glob


def test_perspective_correction():
    for path in tqdm(glob("tests/perspective_correction_tests/*")):
        original = load_image(path)
        cut_image = perspective_correction(original)
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")
        axes[1].imshow(cut_image, cmap="gray")
        axes[1].set_title("Perspective correction")
        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/perspective_correction/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
