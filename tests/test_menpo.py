from tqdm.auto import tqdm
from covidxpert import menpo


def test_menpo():
    paths = [
        "CHNCXR_0001_0",
        "CHNCXR_0002_0",
    ]
    for path in tqdm(paths):
        mask_path = f"tests/menpo_images/{path}_mask.jpg"
        image_path = f"tests/menpo_images/{path}.jpg"
        save_path = f"tests/{path}_points"
        menpo.extract_menpo_points(mask_path, image_path, save_path)
