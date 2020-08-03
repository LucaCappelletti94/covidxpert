"""Module with methods to run the complete pipeline."""
from multiprocessing import cpu_count, Pool
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Dict
from .utils import load_image, remove_artefacts
from .perspective_correction import perspective_correction
from .blur_bbox import blur_bbox
from .counter_rotate import counter_rotate
from .body_cut import get_body_cut


def image_pipeline(
    image_path: str,
    output_path: str,
    blur_bbox_padding: int = 50,
    width: int = 480,
    thumbnail_width: int = 256,
    hardness: float = 0.9,
    artefacts: bool = True,
    retinex: bool = True,
    save_steps: bool = False
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
    width: int = 480,
        The size to resize the image.
    thumbnail_width: int = 256,
        Width to use for the thumbnails during processing.
    hardness: float = 0.9,
        Hardness to use for the body cut.
    artefacts: bool = True,
        Wethever to remove artefacts.
    retinex: bool = True,
        Wethever to apply multiscale retinex at the end of the pipeline.
    save_steps: bool = False,
        Wethever to save the partial steps instead othe processed image.
        This option is useful to debug which parameters are to blaim for
        unexpected pipeline behaviour.
        By default, this is False.
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
        width=thumbnail_width
    )
    # Remove artefacts
    if artefacts:
        image_bbox = remove_artefacts(image_bbox)
    # Cuts the body lower part
    image_body_cut, darken_image_body_cut = get_body_cut(
        image_bbox,
        image_rotated,
        angle,
        simmetry_axis=x,
        width=thumbnail_width,
        hardness=hardness
    )
    directory_name = os.path.dirname(output_path)
    os.makedirs(directory_name, exist_ok=True)
    if not save_steps:
        # Saving image to given path
        plt.savefig(
            image_body_cut if retinex else darken_image_body_cut,
            output_path
        )
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.ravel()

        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Original image")

        axes[1].imshow(image_perspective, cmap="gray")
        axes[1].set_title("Perspective correction")

        axes[2].imshow(image_bbox, cmap="gray")
        axes[2].set_title("Blur BBox image")

        axes[3].imshow(image_rotated, cmap="gray")
        axes[3].set_title("Rotated image")

        axes[4].imshow(darken_image_body_cut, cmap="gray")
        axes[4].set_title("Darkened image")

        axes[5].imshow(image_body_cut, cmap="gray")
        axes[5].set_title("Body cut image")
        [ax.set_axis_off() for ax in axes.ravel()]
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)


def _image_pipeline(kwargs: Dict):
    image_pipeline(**kwargs)


def images_pipeline(
    image_paths: List[str],
    output_paths: List[str],
    blur_bbox_padding: int = 50,
    width: int = 480,
    thumbnail_width: int = 256,
    hardness: float = 0.9,
    artefacts: bool = True,
    retinex: bool = True,
    save_steps: bool = False,
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
    width: int = 480,
        The size to resize the image.
    thumbnail_width: int = 256,
        Width to use for the thumbnails during processing.
    hardness: float = 0.9,
        Hardness to use for the body cut.
    artefacts: bool = True,
        Wethever to remove artefacts.
    retinex: bool = True,
        Wethever to apply multiscale retinex at the end of the pipeline.
    save_steps: bool = False,
        Wethever to save the partial steps instead othe processed image.
        This option is useful to debug which parameters are to blaim for
        unexpected pipeline behaviour.
        By default, this is False.
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
                "givem output paths length ({})."
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
                        blur_bbox_padding=blur_bbox_padding,
                        thumbnail_width=thumbnail_width,
                        hardness=hardness,
                        artefacts=artefacts,
                        retinex=retinex,
                        save_steps=save_steps
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
        p.close()
        p.join()
