def count_from_left_side(mask: np.ndarray):
    counter = 0
    for boolean in mask:
        if boolean:
            counter += 1
        else:
            break
    return counter


def count_from_right_side(mask: np.ndarray):
    return count_from_left_side(np.flip(mask, axis=0))


def build_slice(left: int, right: int, maximum: int):
    return slice(left, maximum if right == 0 else right)


def strip_black(image: np.ndarray, mask: np.ndarray, v_threshold: float, h_threshold: float) -> np.ndarray:
    vertical_mask = mask.mean(axis=1) < v_threshold
    horizzontal_mask = mask.mean(axis=0) < h_threshold

    h_slice = build_slice(
        count_from_left_side(horizzontal_mask),
        -count_from_right_side(horizzontal_mask),
        image.shape[1]
    )
    v_slice = build_slice(
        count_from_left_side(vertical_mask),
        -count_from_right_side(vertical_mask),
        image.shape[0]
    )
    return image[v_slice, h_slice]


def compute_median_threshold(mask: np.ndarray) -> float:
    masked_mask = strip_black(mask, mask, 0, 0)
    v_white_median = np.median(mask.mean(axis=0))
    h_white_median = np.median(mask.mean(axis=1))
    return v_white_median/2, h_white_median/2


def get_blur_mask(image: np.ndarray, padding: int):
    blurred = add_padding(image, padding)
    blurred, _ = remove_artefacts(blurred)
    blurred = apply_mean_blur(blurred)
    blurred = apply_median_threshold(blurred)
    return trim_padding(blurred, padding)


def blur_bbox(image: np.ndarray, padding: int = 50) -> np.ndarray:
    mask = get_blur_mask(image, padding)
    return strip_black(image, mask, *compute_median_threshold(mask))
