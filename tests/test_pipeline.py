import os
from glob import glob
from covidxpert import images_pipeline
import pytest


def test_pipeline():
    """Testing execution of the complete pipeline."""
    image_paths = glob("tests/test_images/*")
    output_paths = [
        "tests/pipeline/{}.jpg".format(
            "".join(os.path.basename(image_path).split(".")[:-1])
        )
        for image_path in image_paths
    ]
    images_pipeline(
        image_paths[:1],
        output_paths[:1],
        verbose=False,
        n_jobs=0
    )
    images_pipeline(
        image_paths,
        output_paths,
        save_steps=True,
        n_jobs=0
    )


def test_pipeline_illegal_arguments():
    image_paths = glob("tests/test_images/*")
    output_paths = []
    with pytest.raises(ValueError):
        images_pipeline(
            image_paths,
            output_paths,
        )
