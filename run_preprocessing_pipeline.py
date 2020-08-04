from notipy_me import Notipy
from covidxpert import images_pipeline

from glob import glob
import pandas as pd

dataset_path = '/data/covidxpert_data/datasets/{}'
output_path = '/data/covidxpert_data/processed/{}'
error_path = '/data/covidxpert_data/error_pipeline/'
width_img = 480
df_index = pd.read_csv(r'/data/covidxpert_data/datasets/normalized_index/df_all_common_cols.csv')
img_path = [dataset_path.format(path) for path in df_index.img_path]
out_path = [output_path.format(path) for path in df_index.img_path]

with Notipy():
    print('start img pipeline')
    images_pipeline(img_path, out_path, error_path, width=width_img)

