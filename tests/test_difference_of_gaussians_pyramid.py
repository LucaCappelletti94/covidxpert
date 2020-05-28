from covidxpert.utils import difference_of_gaussians_pyramid
from covidxpert import load_image

import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import os
import pytest
import numpy as np

def test_difference_of_gaussian_pyramid_wrong_parameters():
    mock_img = np.ndarray(shape=(1,1,3)) 
    with pytest.raises(ValueError):
        difference_of_gaussians_pyramid(mock_img, sigma=-1)

def test_difference_of_gaussian_pyramid_sigma_zero():
    mock_img = np.ndarray(shape=(1,1,3)) 
    difference_of_gaussians_pyramid(mock_img, sigma=0)

def test_difference_of_gaussian_pyramid():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing Difference of Gaussian Pyramid"):
        original = load_image(path)
        background, foreground = difference_of_gaussians_pyramid(original)
        assert isinstance(background, np.ndarray)
        assert isinstance(foreground, np.ndarray)
        fig, axes = plt.subplots(ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(background, cmap="gray")
        axes[1].set_title("Backgroud image")

        axes[2].imshow(foreground, cmap="gray")
        axes[2].set_title("Foreground image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/diff_gaussian_pyramid/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)