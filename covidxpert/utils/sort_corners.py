import numpy as np


def sort_corners(corners: np.ndarray) -> np.ndarray:
    corners_list = corners.tolist()

    top_left_index = np.argmin(np.sum(corners, axis=1))
    top_left = corners_list[top_left_index]

    bottom_right_index = np.argmax(np.sum(corners, axis=1))
    bottom_right = corners_list[bottom_right_index]

    corners_list.remove(bottom_right)
    corners_list.remove(top_left)

    corners = np.array(corners_list)

    bottom_left = corners_list.pop(np.argmin(corners[:, 0]))

    # Getting the botton right corner
    top_right = corners_list.pop()

    return top_left, bottom_left, bottom_right, top_right
