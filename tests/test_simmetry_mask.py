from covidxpert import load_image, perspective_correction
from covidxpert import blur_bbox, get_simmetry_mask
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_simmetry_mask():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing simmetry mask"):
        original = load_image(path)
        cut_image = perspective_correction(original)
        blurred_image = blur_bbox(cut_image)
        mask = get_simmetry_mask(blurred_image)

        fig, axes = plt.subplots(ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(blurred_image, cmap="gray")
        axes[1].set_title("Refined image")

        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Mask")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/simmetry_mask/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
