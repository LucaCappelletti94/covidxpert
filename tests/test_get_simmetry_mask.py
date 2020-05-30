from covidxpert import perspective_correction, blur_bbox
from covidxpert.utils import load_image, remove_artefacts
from covidxpert.simmetry_mask import get_simmetry_mask
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_get_simmetry_mask():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing simmetry mask"):
        original = remove_artefacts(blur_bbox(perspective_correction(load_image(path))))

        fig, axes = plt.subplots(ncols=2)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(get_simmetry_mask(original), cmap="gray")
        axes[1].set_title("Image with borders")

        fig.tight_layout()
        path = f"tests/simmetry_mask/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
