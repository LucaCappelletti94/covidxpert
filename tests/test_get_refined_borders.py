from covidxpert import perspective_correction, blur_bbox
from covidxpert.utils import load_image, get_simmetry_axis, remove_artefacts
from covidxpert.borders import get_refined_borders
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_get_refined_borders():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing refined borders"):
        original = remove_artefacts(blur_bbox(perspective_correction(load_image(path))))

        x = get_simmetry_axis(original)

        fig, axes = plt.subplots(ncols=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(get_refined_borders(original, x), cmap="gray")
        axes[1].set_title("Image with borders")

        fig.tight_layout()
        path = f"tests/refined_borders/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
