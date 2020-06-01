from covidxpert.utils.remove_artefacts import compute_artefacts, remove_artefacts,  fill_small_black_blobs, fill_small_white_blobs
from covidxpert import load_image, perspective_correction

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
        small_black = fill_small_black_blobs(original, fact)
        small_white = fill_small_white_blobs(original, fact)

        assert isinstance(small_black, np.ndarray)
        assert isinstance(small_white, np.ndarray)
        assert small_white.shape == small_black.shape
        fig, axes = plt.subplots(ncols=3)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

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


def test_fill_small_blobs_fact():
    factors = (-20, 0, 20, 40)
    for path in list(tqdm(glob("tests/test_images/*"), desc="Testing Fill Small Blobs"))[:4]:
        original = load_image(path)
        for fact in factors:
            small_black = fill_small_black_blobs(original, fact)
            small_white = fill_small_white_blobs(original, fact)

            assert isinstance(small_black, np.ndarray)
            assert isinstance(small_white, np.ndarray)
            assert small_white.shape == small_black.shape
            fig, axes = plt.subplots(ncols=3)
            axes = axes.ravel()

            axes[0].imshow(original, cmap="gray")
            axes[0].set_title("Original")

            axes[1].imshow(small_black, cmap="gray")
            axes[1].set_title(f"B, {fact}")

            axes[2].imshow(small_white, cmap="gray")
            axes[2].set_title(f"W, {fact}")

            [ax.set_axis_off() for ax in axes.ravel()]
            fig.tight_layout()
            path = f"tests/fill_small_blobs/{os.path.basename(path)}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path.split('.')[0]+'_'+str(fact)+'.jpg')
            plt.close(fig)
