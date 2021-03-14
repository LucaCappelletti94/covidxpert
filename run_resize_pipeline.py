"""
    Script that runs the preprocessing pipeline on input images.
"""
from covidxpert import images_pipeline, resize_images_pipeline

from glob import glob
import pandas as pd
from multiprocessing import cpu_count

if __name__ == "__main__":
    root = "/io/data"
    dataset_path = f'{root}/datasets/{{}}'
    output_path = f'{root}/resized_processed/{{}}'
    error_path = f'{root}/error_pipeline/'
    width = 480
    df_index = pd.read_csv(
        f'{root}/datasets/normalized_index/df_all_common_cols.csv'
    )
    img_path = [dataset_path.format(path) for path in df_index.img_path]
    out_path = [output_path.format(path) for path in df_index.img_path]

    resize_images_pipeline(
        img_path,
        out_path,
        width=width,
        cache=True,
        errors_path=error_path,
        n_jobs=cpu_count()//2  # Using just half of the cores to avoid monopolizing the machine
    )
