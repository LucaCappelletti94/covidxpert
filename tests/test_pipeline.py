import os
from glob import glob
from covidxpert import images_pipeline
import pytest


def test_pipeline():
    """Testing execution of the complete pipeline."""
    image_paths = glob("tests/test_images/*")
    output_paths = [
        "tests/pipeline/{}".format(os.path.basename(image_path))
        for image_path in image_paths
    ]
    images_pipeline(
        image_paths[:1],
        output_paths[:1],
    )
    images_pipeline(
        image_paths,
        output_paths,
        save_steps=True,
    )


def test_pipeline_illegal_arguments():
    image_paths = glob("tests/test_images/*")
    output_paths = []
    with pytest.raises(ValueError):
        images_pipeline(
            image_paths,
            output_paths,
        )
