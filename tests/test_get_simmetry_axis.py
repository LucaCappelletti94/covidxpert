from covidxpert import perspective_correction, blur_bbox
from covidxpert.utils import load_image, get_simmetry_axis, remove_artefacts
from tqdm.auto import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os


def test_get_simmetry_axis():
    for path in tqdm(glob("tests/test_images/*"), desc="Testing simmetry axis"):
        original = remove_artefacts(blur_bbox(perspective_correction(load_image(path))))

        fig, axes = plt.subplots(ncols=1)

        axes.imshow(original, cmap="gray")
        axes.axvline(get_simmetry_axis(original))
        axes.axvline(original.shape[1]//2)
        axes.set_title("Simmetry axis and actual center of image.")

        fig.tight_layout()
        path = f"tests/simmetry_axis/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
