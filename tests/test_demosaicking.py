import os
from glob import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from covidxpert.utils import load_image, get_thumbnail, demosaicking
from covidxpert.perspective_correction import perspective_correction
from covidxpert.blur_bbox import blur_bbox
from covidxpert.counter_rotate import counter_rotate
from covidxpert.body_cut import get_body_cut


def test_demosaicking():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing demosaicking"):
        original = load_image(path)
        image_demo = demosaicking(original, 'menon')

        image_perspective = perspective_correction(image_demo)
        image_perspective = get_thumbnail(image_perspective, width=1024)
        image_bbox = blur_bbox(image_perspective)
        image_rotated, angle, x = counter_rotate(image_bbox)
        image_body_cut, _ = get_body_cut(
            image_bbox,
            image_rotated,
            angle,
            simmetry_axis=x
        )
        image_bbox = blur_bbox(image_body_cut)

        fig, axes = plt.subplots(figsize=(10, 5), ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(image_demo, cmap="gray")
        axes[1].set_title("Demo Image")

        axes[2].imshow(image_bbox, cmap="gray")
        axes[2].set_title("Bounding box")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/demosaicking/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
