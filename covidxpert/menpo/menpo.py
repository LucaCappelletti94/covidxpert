import numpy as np
from typing import Tuple
import cv2
import bezier
from menpo.shape import PointCloud
from menpo.io import export_landmark_file
from ..perspective_correction.get_corners import get_cardinal_corner_points
from ..utils import normalize_image, get_thumbnail, load_image
from ..body_cut import get_body_cut
from ..perspective_correction import perspective_correction
from ..blur_bbox import blur_bbox
from ..counter_rotate import counter_rotate


def get_mask_lungs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return tuple with the lungs masks."""

    _, output, stats, _ = cv2.connectedComponentsWithStats( # pylint: disable=no-member
        mask, connectivity=8
    )

    one = output == 1
    two = output == 2

    one_x = np.mean(np.where(one), axis=1)[1]
    two_x = np.mean(np.where(two), axis=1)[1]

    one = normalize_image(one)
    two = normalize_image(two)

    if one_x < two_x:
        return one, two
    return two, one


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def sort_contour(contour, start: Tuple[int, int]):
    sorted_contour = []
    total_points = len(contour)
    unique_points = set()
    while len(sorted_contour) != total_points:
        point_arg = closest_node(start, contour)
        start = contour[point_arg]
        del contour[point_arg]
        if tuple(start) not in unique_points:
            sorted_contour.append(start)
            unique_points.add(tuple(start))
    return sorted_contour


def get_corners(mask):
    contour = cv2.goodFeaturesToTrack( # pylint: disable=no-member
        image=mask,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=20
    ).reshape(-1, 2).astype(int)

    corners = get_cardinal_corner_points(contour)
    sorted_contours = sort_contour(contour.tolist(), corners[0])

    return corners, sorted_contours


def split_contours(corners, contours):
    parts = [
        [] for _ in corners
    ]

    counter = -1
    for p in contours:
        for corner in corners:
            if (corner == p).all():
                counter += 1
        parts[counter].append(p)

    parts = [
        np.array(part)
        for part in parts
    ]

    x, y = list(zip(*[
        part.mean(axis=0).T
        for part in parts
    ]))

    x = list(x)
    y = list(y)

    top_part_index = np.argmin(y)
    bottom_part_index = np.argmax(y)

    x[bottom_part_index] = np.mean(x)
    x[top_part_index] = np.mean(x)

    left_part_index = np.argmin(x)
    right_part_index = np.argmax(x)

    top_part = parts[top_part_index]
    bottom_part = parts[bottom_part_index]
    left_part = parts[left_part_index]
    right_part = parts[right_part_index]

    # Sort left part from top to bottom
    left_part = np.array(sorted(left_part, key=lambda e: e[1], reverse=False))
    # Sort bottom part from left to right
    bottom_part = np.array(sorted(bottom_part, key=lambda e: e[0], reverse=False))
    # Sort right part from bottom to top
    right_part = np.array(sorted(right_part, key=lambda e: e[1], reverse=True))
    # Sort top part from right to left
    top_part = np.array(sorted(top_part, key=lambda e: e[0], reverse=True))

    return (
        left_part,
        bottom_part,
        right_part,
        top_part
    )


def sample_points(points, k: int = 10) -> np.ndarray:
    curve = bezier.Curve.from_nodes(points.T)
    return curve.evaluate_multi(np.linspace(0, 1, k))


def sampled_to_points(points_sampled):
    return [
        tuple(pt)
        for sample in points_sampled
        for pt in sample.T
    ]


def convert_to_point_cloud(left_sampled, right_sampled) -> PointCloud:
    return PointCloud(
        sampled_to_points(left_sampled) + sampled_to_points(right_sampled)
    )


def extract_menpo_points(mask_path, image_path, save_path):
    mask = get_thumbnail(load_image(mask_path), 1024)
    image = get_thumbnail(load_image(image_path), 1024)

    image, others = perspective_correction(image, others=[mask])
    image, others = blur_bbox(image, others=others)
    rotated, angle, x, others = counter_rotate(image, others=others)
    body_cut, dark_body_cut, others = get_body_cut(image, rotated, angle, x, others=others)

    mask = others[0]

    left, right = get_mask_lungs(mask)

    left_corners, left_contours = get_corners(left)
    right_corners, right_contours = get_corners(right)

    left_parts = split_contours(left_corners, left_contours)
    right_parts = split_contours(right_corners, right_contours)

    left_sampled = [
        sample_points(part, 5)
        for part in left_parts
    ]

    right_sampled = [
        sample_points(part, 5)
        for part in right_parts
    ]

    point_cloud = convert_to_point_cloud(left_sampled, right_sampled)

    export_landmark_file(point_cloud, f"{save_path}.pts", overwrite=True)
