import os
from glob import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from covidxpert import load_image, perspective_correction, blur_bbox, get_body_cut, counter_rotate


def test_pipeline():
    for path in tqdm(glob("tests/test_images/*"), desc="Test pipeline"):
        original = load_image(path)
        image_perspective = perspective_correction(original)
        image_bbox = blur_bbox(image_perspective)
        image_rotated, angle, x = counter_rotate(image_bbox)
        image_body_cut, _ = get_body_cut(image_bbox, image_rotated, angle, x)

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(image_perspective, cmap="gray")
        axes[1].set_title("Perspective correction")

        axes[2].imshow(image_bbox, cmap="gray")
        axes[2].set_title("Blur BBox image")

        axes[3].imshow(image_rotated, cmap="gray")
        axes[3].set_title("Rotated image")

        axes[4].imshow(image_body_cut, cmap="gray")
        axes[4].set_title("Body cut image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/pipeline/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
