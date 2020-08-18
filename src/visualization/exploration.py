import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def grid_show(
    sample: pd.DataFrame,
    base_dir: str,
    width: int,
    height: int,
    columns: int,
    rows: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(width, height))
    for i in range(1, columns * rows + 1):
        img_path = Path(base_dir, sample.loc[i - 1, "image_name"] + ".jpg")
        image = Image.open(img_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
    return fig


def grid_augmentations_show(
    sample: list, width: int, height: int, columns: int, rows: int,
):
    fig = plt.figure(figsize=(width, height))
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        image = sample[i - 1].numpy()
        plt.imshow(np.transpose(image, (1, 2, 0)), interpolation="nearest")
    return fig
