from tqdm.auto import tqdm
from covidxpert.menpo import extract_menpo_points


def test_menpo():
    paths = [
        "CHNCXR_0001_0",
        "CHNCXR_0002_0",
    ]
    for path in tqdm(paths):
        mask_path = f"tests/menpo_images/{path}_mask.jpg"
        image_path = f"tests/menpo_images/{path}.jpg"
        save_image_path = f"tests/menpo_images/{path}_processed.png"
        save_points_path = f"tests/menpo_images/{path}_points.pts"
        extract_menpo_points(mask_path, image_path, save_image_path, save_points_path)
