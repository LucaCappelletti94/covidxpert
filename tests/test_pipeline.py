from covidxpert import load_image, perspective_correction, blur_bbox
from covidxpert.perspective_correction.get_masked_image import get_masked_image
from covidxpert.utils import histogram_based_vertical_thresholding
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_pipeline():
    for path in tqdm(glob("tests/test_images/*"), desc="Test pipeline"):
        original = load_image(path)
        image_perspective = perspective_correction(original)
        image_bbox = blur_bbox(image_perspective)
        image_rotated = image_bbox
        image_body_cut = histogram_based_vertical_thresholding(image_rotated, get_masked_image(image_rotated))
        image_horizontal_cut = image_body_cut

        fig, axes = plt.subplots(nrows=2, ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(image_perspective, cmap="gray")
        axes[1].set_title("Perspective correction")

        axes[2].imshow(image_bbox, cmap="gray")
        axes[2].set_title("Bounding box image")

        axes[3].imshow(image_rotated, cmap="gray")
        axes[3].set_title("Rotated image (WIP)")

        axes[4].imshow(image_body_cut, cmap="gray")
        axes[4].set_title("Body cut image")

        axes[5].imshow(image_horizontal_cut, cmap="gray")
        axes[5].set_title("Horizontal cut image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/pipeline/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
