import os
from glob import glob
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import colour
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
colour.utilities.filter_warnings()
from covidxpert.utils import load_image, get_thumbnail
from covidxpert.perspective_correction import perspective_correction
from covidxpert.blur_bbox import blur_bbox
from covidxpert.counter_rotate import counter_rotate
from covidxpert.body_cut import get_body_cut


def pipeline(input):
    image_perspective = perspective_correction(input)
    image_perspective = get_thumbnail(image_perspective, width=1024)
    image_bbox = blur_bbox(image_perspective)
    image_rotated, angle, x = counter_rotate(image_bbox)
    image_body_cut, _ = get_body_cut(
        image_bbox,
        image_rotated,
        angle,
        simmetry_axis=x
    )
    return blur_bbox(image_body_cut)


def test_demosaicking():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing demosaicking"):
        original = colour.io.read_image(path)

        if len(original.shape) == 2:
            image = cv2.merge((original, original, original))
        else:
            image = original

        CFA = mosaicing_CFA_Bayer(image)
        demo_bilinear = ((colour.cctf_encoding(demosaicing_CFA_Bayer_bilinear(CFA)))*255).astype(np.uint8)
        demo_malvar = ((colour.cctf_encoding(demosaicing_CFA_Bayer_Malvar2004(CFA)))*255).astype(np.uint8)
        demo_menon = ((colour.cctf_encoding(demosaicing_CFA_Bayer_Menon2007(CFA)))*255).astype(np.uint8)

        demosaicing_bilinear_body_cut = pipeline(
            cv2.cvtColor(demo_bilinear, cv2.COLOR_RGB2GRAY))
        demosaicing_malvar_body_cut = pipeline(
            cv2.cvtColor(demo_malvar, cv2.COLOR_RGB2GRAY))
        demosaicing_menon_body_cut = pipeline(
            cv2.cvtColor(demo_menon, cv2.COLOR_RGB2GRAY))

        original_classic = load_image(path)
        image_body_cut = pipeline(original_classic)

        fig, axes = plt.subplots(figsize=(15, 15), ncols=2, nrows=5)
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(CFA, cmap="gray")
        axes[1].set_title("CFA")

        axes[2].imshow(demo_bilinear, cmap="gray")
        axes[2].set_title("Demo Bilinear")

        axes[3].imshow(demosaicing_bilinear_body_cut, cmap="gray")
        axes[3].set_title("Bilinear body cut")

        axes[4].imshow(demo_malvar, cmap="gray")
        axes[4].set_title("Demo Malvar")

        axes[5].imshow(demosaicing_malvar_body_cut, cmap="gray")
        axes[5].set_title("Malvar body cut")

        axes[6].imshow(demo_menon, cmap="gray")
        axes[6].set_title("Demo Menon")

        axes[7].imshow(demosaicing_menon_body_cut, cmap="gray")
        axes[7].set_title("Menon body cut")

        axes[8].imshow(original_classic, cmap="gray")
        axes[8].set_title("Original read")

        axes[9].imshow(image_body_cut, cmap="gray")
        axes[9].set_title("Original body cut")

        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        path = f"tests/demosaicking/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)