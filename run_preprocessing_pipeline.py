"""
    Script that runs the preprocessing pipeline on input images.
"""
from notipy_me import Notipy
from covidxpert import images_pipeline

from glob import glob
import pandas as pd
import multiprocessing

root = '/data/covidxpert_data'
dataset_path = f'{root}/datasets/{{}}'
output_path = f'{root}/processed/{{}}.jpg'
error_path = f'{root}/error_pipeline'

# Width to which the images are resized initially
initial_witdh = 1024

# Width to which the output images are resized 
width_img = 480

if __name__ == "__main__":
    
    # Load dataset images path
    df_index = pd.read_csv(f'{root}/datasets/normalized_index/df_all_common_cols.csv')
    
    #Format images path
    img_path = [dataset_path.format(path) for path in df_index.img_path]
    out_path = [output_path.format("".join(path.split('.')[:-1])) for path in df_index.img_path]

    
    with Notipy(task_name="Image pipeline preprocessing"):
        images_pipeline(img_path, out_path, 
                        errors_path=error_path, 
                        initial_width=initial_witdh,
                        width=width_img,
                        n_jobs=multiprocessing.cpu_count()//2)

