from covidxpert import load_image, perspective_correction
from covidxpert.perspective_correction.get_corners import get_corners
from covidxpert.perspective_correction.add_padding import add_padding
from covidxpert.perspective_correction.get_masked_image import get_masked_image
from covidxpert.perspective_correction.cut_bounding_box import cut_bounding_box
from covidxpert.perspective_correction.perspective_correction import get_new_cardinals
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import os
from glob import glob


def test_perspective_correction():
    for path in tqdm(glob("tests/test_images/*")):
        original = load_image(path)
        cut_image = perspective_correction(original)
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.ravel()
        
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")
        padded = add_padding(original)
        corners, requires_correction, score = get_corners(padded)

        axes[1].imshow(get_masked_image(padded), cmap="gray")
        axes[1].scatter(*corners.T, marker='.')
        axes[1].set_title("Identified corners ({0:0.4})".format(score))

        if requires_correction:
            padded = cut_bounding_box(padded, corners)
            corners -= corners.min(axis=0)
            new_corners = get_new_cardinals(padded)

            axes[2].imshow(padded, cmap="gray")
            axes[2].scatter(*corners.T, marker='.')
            axes[2].scatter(*new_corners.T, marker='.')
            axes[2].set_title("Repositioned corners")
            
        axes[3].imshow(cut_image, cmap="gray")
        axes[3].set_title("Perspective correction")
        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/perspective_correction/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
