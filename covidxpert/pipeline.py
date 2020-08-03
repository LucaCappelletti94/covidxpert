"""Module with methods to run the complete pipelie."""
from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Dict
from .utils import load_image
from .perspective_correction import perspective_correction
from .blur_bbox import blur_bbox
from .counter_rotate import counter_rotate
from .body_cut import get_body_cut


def image_pipeline(
    image_path: str,
    output_path: str,
    blur_bbox_padding: int = 50,
    image_width: int = 256,
    hardness: float = 0.9
):
    """Executes complete pipeline on given image.

    Parameters
    ---------------------------
    image_path: str,
        Path from where to load the given image.
    output_path: str,
        Path where to save the processed image.
    blur_bbox_padding: int = 50,
        The padding to use around the blur bbox cut.
    image_width: int = 256,
        Image width for processed image.
    hardness: float = 0.9,
        Hardness to use for the body cut.
    """
    # Loading the image.
    original = load_image(image_path)

    # Executes perspective correction
    image_perspective = perspective_correction(original)

    # Executes blur bbox cut
    image_bbox = blur_bbox(
        image_perspective,
        padding=blur_bbox_padding
    )
    # Determines optimal counter rotation
    image_rotated, angle, x = counter_rotate(
        image_bbox,
        width=image_width
    )
    # Cuts the body lower part
    image_body_cut, _ = get_body_cut(
        image_bbox,
        image_rotated,
        angle,
        simmetry_axis=x,
        width=image_width,
        hardness=hardness
    )
    # Saving image to given path
    plt.savefig(image_body_cut, output_path)


def _image_pipeline(kwargs: Dict):
    image_pipeline(**kwargs)


def images_pipeline(
    image_paths: List[str],
    output_paths: List[str],
    blur_bbox_padding: int = 50,
    image_width: int = 256,
    hardness: float = 0.9,
    n_jobs: int = None,
    verbose: bool = True
):
    """Executes complete pipeline on given image.

    Parameters
    ---------------------------
    image_path: str,
        Path from where to load the given image.
    output_path: str,
        Path where to save the processed image.
    blur_bbox_padding: int = 50,
        The padding to use around the blur bbox cut.
    image_width: int = 256,
        Image width for processed image.
    hardness: float = 0.9,
        Hardness to use for the body cut.
    n_jobs: int = None,
        Number of jobs to use for the processing task.
        If given value is None, the number of available CPUs is used.
    verbose: bool = True,
        Wethever to show the loading bar.

    Raises
    -------------------------
    ValueError,
        If given image paths length does not match given output paths length.
    """

    if len(image_paths) != len(output_paths):
        raise ValueError(
            (
                "Given image paths length ({}) does not match the length of "
                "givem output paths length."
            ).format(
                len(image_paths), len(output_paths)
            )
        )

    if n_jobs is None:
        n_jobs = cpu_count()

    n_jobs = min(len(image_paths), n_jobs)

    with Pool(n_jobs) as p:
        list(tqdm(
            p.imap(
                _image_pipeline,
                (
                    dict(
                        image_path=image_path,
                        output_path=output_path,
                        blur_bbox_padding=50,
                        image_width=256,
                        hardness=0.75
                    )
                    for image_path, output_path in zip(
                        image_paths,
                        output_paths
                    )
                )
            ),
            desc="Processing images",
            total=len(image_paths),
            disable=not verbose
        ))
