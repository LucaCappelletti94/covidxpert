import silence_tensorflow.auto
from covidxpert import load_image, LungSegmenter, perspective_correction
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import os


def test_lung_segmentation():
    segmenter = LungSegmenter()
    for path in tqdm(glob("tests/test_images/*")):
        original = load_image(path)
        cut_image = perspective_correction(original)
        segmented = segmenter.predict(cut_image)
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")
        axes[1].imshow(cut_image, cmap="gray")
        axes[1].set_title("Perspective correction")
        axes[2].imshow(segmented, cmap="gray")
        axes[2].set_title("Lung segmentation")
        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/lung_segmentation/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
