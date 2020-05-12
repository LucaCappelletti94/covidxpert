from covidxpert import load_image, LungSegmenter, perspective_correction, blur_bbox
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import os
import cv2


def test_lung_segmentation():
    segmenter = LungSegmenter()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))
    for path in tqdm(glob("tests/test_images/*"), desc="Test lung segmentation"):
        original = load_image(path)
        cut_image = perspective_correction(original)
        cut_image = blur_bbox(cut_image)
        cut_image = clahe.apply(cut_image)
        segmented = segmenter.predict(cut_image)
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")
        axes[1].imshow(cut_image, cmap="gray")
        axes[1].set_title("Cut and reshape")
        axes[2].imshow(segmented, cmap="gray")
        axes[2].set_title("Lung segmentation")
        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/lung_segmentation/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
