from notipy_me import Notipy
from covidxpert import images_pipeline

from glob import glob
import pandas as pd

root = "~"
dataset_path = f'{root}/datasets/{{}}'
output_path = f'{root}/processed/{{}}'
error_path = f'{root}/error_pipeline/'
width = 480
initial_width = 1024
df_index = pd.read_csv(
    f'{root}/datasets/normalized_index/df_all_common_cols.csv'
)
img_path = [dataset_path.format(path) for path in df_index.img_path]
out_path = [output_path.format(path) for path in df_index.img_path]

with Notipy():
    images_pipeline(
        img_path,
        out_path,
        width=width,
        initial_width=initial_width,
        cache=True,
        errors_path=error_path
    )
