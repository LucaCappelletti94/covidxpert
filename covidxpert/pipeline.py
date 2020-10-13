"""Module with methods to run the complete pipeline."""
from multiprocessing import cpu_count, Pool
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import List, Dict
import cv2
import compress_json
from colorcorrect.algorithm import automatic_color_equalization as ace
from .utils import load_image, get_thumbnail
from .perspective_correction import perspective_correction
from .blur_bbox import blur_bbox
from .counter_rotate import counter_rotate
from .body_cut import get_body_cut
import traceback


def image_pipeline(
    image_path: str,
    output_path: str = None,
    blur_bbox_padding: int = 50,
    width: int = 480,
    initial_width: int = 1024,
    thumbnail_width: int = 256,
    hardness: float = 0.7,
    save_steps: bool = False,
    cache: bool = False,
    errors_path: str = None,
):
    """Executes complete pipeline on given image.

    Parameters
    ---------------------------
    image_path: str,
        Path from where to load the given image.
    output_path: str = None,
        Path where to save the processed image.
        Use None to not save the image.
    blur_bbox_padding: int = 50,
        The padding to use around the blur bbox cut.
    width: int = 480,
        The size to resize the image.
    initial_width: int = 1024,
        The initial resize width.
    thumbnail_width: int = 256,
        Width to use for the thumbnails during processing.
    hardness: float = 0.7,
        Hardness to use for the body cut.
    save_steps: bool = False,
        Wethever to save the partial steps instead othe processed image.
        This option is useful to debug which parameters are to blaim for
        unexpected pipeline behaviour.
        By default, this is False.
    cache: bool = False,
        Wethever to skip processing an image if it was already processed.
    errors_path: str = None,
        Path where to store the error info.
        Use None to not log the errors and raise them.
    """
    try:
        # Check if we have already this image caches
        if cache and output_path is not None and os.path.exists(output_path):
            # If this is the case we skip this image.
            return None

        # Loading the image.
        original = load_image(image_path)

        # Executes perspective correction
        image_perspective = perspective_correction(original)

        # Computing thumbnail
        image_perspective = get_thumbnail(
            image_perspective, width=initial_width)

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

        # Cuts the body lower part
        image_body_cut, _ = get_body_cut(
            image_bbox,
            image_rotated,
            angle,
            simmetry_axis=x,
            width=thumbnail_width,
            hardness=hardness
        )

        # Executes secondary blur bbox cut
        image_body_cut = blur_bbox(
            image_body_cut,
            padding=blur_bbox_padding
        )

        # Final result
        final_result = cv2.cvtColor(  # pylint: disable=no-member
            ace(
                cv2.cvtColor(  # pylint: disable=no-member
                    image_body_cut,
                    cv2.COLOR_GRAY2RGB  # pylint: disable=no-member
                ),
                slope=10
            ),
            cv2.COLOR_RGB2GRAY  # pylint: disable=no-member
        )
    # If the user hit a keyboard interrupt we just stop.
    except KeyboardInterrupt as e:
        raise e
    # Otherwise we optionally write to disk to encountered exception.
    except Exception as e:
        if errors_path is None:
            raise e
        os.makedirs(errors_path, exist_ok=True)
        compress_json.dump(
            {
                "image-path": image_path,
                "text": str(e),
                "class": str(type(e)),
                "traceback": " ".join(
                    traceback.format_exc().splitlines()
                )
            },
            "{}/{}.json".format(
                errors_path,
                os.path.basename(image_path).split(".")[0]
            ),
            json_kwargs=dict(
                indent=4
            )
        )
        return None

    if output_path is not None:
        directory_name = os.path.dirname(output_path)
        os.makedirs(directory_name, exist_ok=True)

    thumb = get_thumbnail(image_body_cut, width=width)

    if not save_steps:
        if output_path is not None:
            # Saving image to given path
            cv2.imwrite(  # pylint: disable=no-member
                output_path,
                # Resize given image
                thumb
            )
        return thumb

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

    axes[4].imshow(image_body_cut, cmap="gray")
    axes[4].set_title("Body cut image")

    axes[5].imshow(final_result, cmap="gray")
    axes[5].set_title("ACE")

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
    initial_width: int = 1024,
    thumbnail_width: int = 256,
    hardness: float = 0.7,
    save_steps: bool = False,
    cache: bool = False,
    errors_path: str = None,
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
    initial_width: int = 1024,
        The initial resize width.
    thumbnail_width: int = 256,
        Width to use for the thumbnails during processing.
    hardness: float = 0.7,
        Hardness to use for the body cut.
    save_steps: bool = False,
        Wethever to save the partial steps instead othe processed image.
        This option is useful to debug which parameters are to blaim for
        unexpected pipeline behaviour.
        By default, this is False.
    cache: bool = False,
        Wethever to skip processing an image if it was already processed.
    errors_path: str = None,
        Path where to store the error info.
        Use None to not log the errors and raise them.
    n_jobs: int = None,
        Number of jobs to use for the processing task.
        If given value is None, the number of available CPUs is used.
        If 0 is used, we do not use multiprocessing.
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
                "given output paths length ({})."
            ).format(
                len(image_paths), len(output_paths)
            )
        )

    if n_jobs is None:
        n_jobs = cpu_count()

    n_jobs = min(len(image_paths), n_jobs)

    tasks = (
        dict(
            image_path=image_path,
            output_path=".".join(output_path.split(".")[:-1]) + ".jpg",
            blur_bbox_padding=blur_bbox_padding,
            thumbnail_width=thumbnail_width,
            hardness=hardness,
            save_steps=save_steps,
            cache=cache,
            errors_path=errors_path
        )
        for image_path, output_path in zip(
            image_paths,
            output_paths
        )
    )

    if n_jobs == 0:
        for kwargs in tqdm(
            tasks,
            desc="Processing images",
            total=len(image_paths),
            disable=not verbose
        ):
            _image_pipeline(kwargs)
    else:
        with Pool(n_jobs) as p:
            list(tqdm(
                p.imap(_image_pipeline, tasks),
                desc="Processing images",
                total=len(image_paths),
                disable=not verbose
            ))
            p.close()
            p.join()
