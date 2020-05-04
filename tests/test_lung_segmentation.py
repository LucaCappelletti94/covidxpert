from covidxpert.utils import load_image
from covidxpert.lung_segmentation import LungSegmenter
from covidxpert import perspective_correction
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm


def test_perspective_correction():
    for i, path in enumerate(tqdm(glob("sample_dataset/*")[:5])):
        segmenter = LungSegmenter(((64, 64, 1)))
        # original = load_image(path)
        # cut_image = perspective_correction(original)
        # segmented = segmenter.predict(cut_image)
        # fig, axes = plt.subplots(ncols=3)
        # axes[0].imshow(original)
        # axes[1].imshow(cut_image)
        # axes[2].imshow(segmented)
        # fig.savefig(f"{i}.jpg")
        # fig.close()
