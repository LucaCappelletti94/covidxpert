def get_kernel_size(image: np.ndarray) -> int:
    """Return the kernel size based on the image size.

    Parameters
    ------------------
    image: np.ndarray,
        The image for which to obtain the kernel.

    Returns
    ------------------
    The kernel size for the image.
    """
    kernel_size = int(min(image.shape) / 5)

    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size
