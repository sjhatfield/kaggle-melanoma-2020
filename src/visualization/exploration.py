import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def grid_show(
    sample: pd.DataFrame,
    base_dir: str,
    width: int,
    height: int,
    columns: int,
    rows: int,
):
    fig = plt.figure(figsize=(width, height))
    for i in range(1, columns * rows + 1):
        img_path = Path(base_dir, sample.loc[i - 1, "image_name"] + ".jpg")
        image = Image.open(img_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
