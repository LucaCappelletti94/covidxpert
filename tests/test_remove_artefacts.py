from covidxpert.utils.remove_artefacts import compute_artefacts, remove_artefacts,  fill_small_black_blobs, fill_small_white_blobs
from covidxpert import load_image, perspective_correction

from covidxpert.utils.normalize_image import normalize_image
from covidxpert.utils.median_mask import median_mask


import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import os
import pytest
import numpy as np


def test_remove_artefacts():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing Remove Artefacts"):
        original = load_image(path)
        corrected = perspective_correction(original)
        artefacts = compute_artefacts(corrected)
        cleared_image = remove_artefacts(corrected)

        assert isinstance(corrected, np.ndarray)
        assert isinstance(artefacts, np.ndarray)
        assert corrected.shape == cleared_image.shape == artefacts.shape
        fig, axes = plt.subplots(ncols=4)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(corrected, cmap="gray")
        axes[1].set_title("Corrected image")

        axes[2].imshow(artefacts, cmap="gray")
        axes[2].set_title("Artefacts image")

        axes[3].imshow(cleared_image, cmap="gray")
        axes[3].set_title("Cleared image")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/remove_artefacts/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def test_fill_small_blobs():
    fact = 20
    for path in tqdm(glob("tests/test_images/*"), desc="Testing Fill Small Blobs"):
        original = load_image(path)
        mask = normalize_image(median_mask(original))

        small_black = fill_small_black_blobs(mask, fact)
        small_white = fill_small_white_blobs(mask, fact)

        assert isinstance(small_black, np.ndarray)
        assert isinstance(small_white, np.ndarray)
        assert small_white.shape == small_black.shape
        fig, axes = plt.subplots(ncols=3)
        axes = axes.ravel()

        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title("Mask image")

        axes[1].imshow(small_black, cmap="gray")
        axes[1].set_title(f"Small black, factor {fact}")

        axes[2].imshow(small_white, cmap="gray")
        axes[2].set_title(f"Small white, factor {fact}")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/fill_small_blobs/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def test_fill_small_blobs_fact_0():
    mock_mask = np.ndarray(shape=(1, 1, 3))
    with pytest.raises(ValueError):
        fill_small_black_blobs(mock_mask, factor=0)
    with pytest.raises(ValueError):
        fill_small_white_blobs(mock_mask, factor=0)


def test_fill_small_blobs_fact():
    n_sample = 4
    nfig_row = 3 
    factors = (-1, 1.5, 
                2, 3, 5,
                10, 20, 100)
    map_types = {'Black': fill_small_black_blobs, 'White': fill_small_white_blobs}

    for path in list(tqdm(glob("tests/test_images/*"), desc="Testing Fill Small Blobs Fact"))[:n_sample]:
        original = load_image(path)
        #mask = normalize_image(median_mask(original))

        for t in map_types:
            fig, axes = plt.subplots(nrows=(len(factors)+1)//nfig_row, ncols=nfig_row, figsize=(15,10))
            #axes = axes.ravel()
            axes[0][0].imshow(normalize_image(median_mask(original)), cmap="gray")
            axes[0][0].set_title("Mask")
            for j,fact in enumerate(factors, 1):
                filled_mask = map_types[t](normalize_image(median_mask(original)), fact)

                axes[j//nfig_row][j%nfig_row].imshow(filled_mask, cmap="gray")
                axes[j//nfig_row][j%nfig_row].set_title(f"{t}, {fact}")

                [ax.set_axis_off() for ax in axes.ravel()]
                fig.tight_layout()

            path = f"tests/fill_small_blobs/{os.path.basename(path)}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(f"{path.split('.')[0]}_{t}_factors.jpg", bbox_inches = 'tight')
            plt.close(fig)

def test_fill_small_blobs_shape():
    fact=20
    for path in list(tqdm(glob("tests/test_images/*"), desc="Testing Fill Small Blobs Shape")):
        original = load_image(path)
        mask = normalize_image(median_mask(original))
        small_black = fill_small_black_blobs(mask, fact)
        small_white = fill_small_white_blobs(mask, fact)

        assert isinstance(small_black, np.ndarray)
        assert isinstance(small_white, np.ndarray)
        assert small_white.shape == small_black.shape