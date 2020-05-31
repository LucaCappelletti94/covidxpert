from tqdm.auto import tqdm
from glob import glob
from covidxpert import blur_bbox
from covidxpert import load_image, perspective_correction
from covidxpert.simmetry_mask import get_simmetry_mask
from covidxpert.utils import remove_artefacts, histogram_based_thresholding
import os
import matplotlib.pyplot as plt


def test_histogram_based_thresholding():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing histogram cut"):
        original = remove_artefacts(blur_bbox(perspective_correction(load_image(path))))
        mask = get_simmetry_mask(original)

        mask_slice, image_slice = histogram_based_thresholding(original, mask)
        mask_slice2, image_slice2 = histogram_based_thresholding(original, mask, 0.8)
        mask_slice3, image_slice3 = histogram_based_thresholding(original, mask, 0.4)

        fig, axes = plt.subplots(ncols=2, nrows=4)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")

        axes[2].imshow(mask_slice, cmap="gray")
        axes[2].set_title("Mask slice (percentage=0.6)")

        axes[3].imshow(image_slice, cmap="gray")
        axes[3].set_title("Image slice (percentage=0.6)")

        axes[4].imshow(mask_slice2, cmap="gray")
        axes[4].set_title("Mask slice (percentage=0.8)")

        axes[5].imshow(image_slice2, cmap="gray")
        axes[5].set_title("Image slice (percentage=0.8)")

        axes[6].imshow(mask_slice3, cmap="gray")
        axes[6].set_title("Mask slice (percentage=0.4)")

        axes[7].imshow(image_slice3, cmap="gray")
        axes[7].set_title("Image slice (percentage=0.4)")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/histogram_based_thresholding/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
