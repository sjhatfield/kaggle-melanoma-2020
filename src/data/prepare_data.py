import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
import torch


class UniformAugment:
    """
    Contains a number of image augmentations that may be chosen
    uniformly at random during training and testing. The default
    fill color is grey
    :param operation: number of augmentations to be performed
    :param fillcolor: RGB tuple
    """

    def __init__(self, operations: int = 2, fillcolor: tuple = (128, 128, 128)):
        self.operations = operations
        # These are the ranges of possible values for each of the augmentations
        # A range of [0, 0] means the augmentation has no value, it just happens
        self.augs = {
            "shearX": [-0.3, 0.3],
            "shearY": [-0.3, 0.3],
            "translateX": [-0.45, 0.45],
            "translateY": [-0.45, 0.45],
            "rotate": [-30, 30],
            "autocontrast": [0, 0],
            "invert": [0, 0],
            "equalize": [0, 0],
            "solarize": [0, 256],
            "posterize": [4, 8],
            "contrast": [0.1, 1.9],
            "color": [0.1, 1.9],
            "brightness": [0.1, 1.9],
            "sharpness": [0.1, 1.9],
            "cutout": [0, 0.2],
        }

        def rotate_with_fill(img: PIL.Image, degrees: float) -> PIL.Image:
            """
            :param img: the image in PIL format
            :param degrees: degrees counter clockwise to be rotated
            """
            rot = img.convert("RGBA").rotate(degrees)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        def cutout(img, magnitude: float, fillcolor: tuple) -> PIL.Image:
            """
            :param magnitude: the  percentage as a decimal that will be cut out
            :param fillcolor: RGB tuple
            """
            img = img.copy()
            w, h = img.size
            v = w * magnitude
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)
            x0 = int(max(0, x0 - v / 2.0))
            y0 = int(max(0, y0 - v / 2.0))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)
            xy = (x0, y0, x1, y1)
            ImageDraw.Draw(img).rectangle(xy, fillcolor)
            return img

        """
        These are all the possible augmentations of the image,
        most of which are straight from the PIL package
        """
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude, 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, int(magnitude)),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                magnitude
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                magnitude
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                magnitude
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            "cutout": lambda img, magnitude: cutout(
                img, magnitude, fillcolor=fillcolor
            ),
        }

    def __call__(self, img: PIL.Image) -> PIL.Image:
        """
        Select a random sample of augmentations where 
        self.operations determines how many and perform them
        on the image
        """
        operations = random.sample(list(self.augs.items()), self.operations)
        for operation in operations:
            augmentation, range = operation
            # Uniformly select value from range of augmentations
            magnitude = random.uniform(range[0], range[1])
            probability = random.random()
            # Perform augmentation uniformly at random
            if random.random() < probability:
                img = self.func[augmentation](img, magnitude)
        return img


class ImageTransform:
    """
    Image transformations to be done which depend on what
    phase of learning is taking place. Normalization happens
    first
    """

    def __init__(
        self,
        resize: int,
        uniform_augment: bool,
        mean: tuple = (0.485, 0.456, 0.406),  # ImageNet
        std: tuple = (0.229, 0.224, 0.225),  # ImageNet
        train: bool = True,
    ):
        """
        :param resize: integer giving the square side length in pixels of resized image
        :param uniform_augment: bool indicating where to use uniform augmentations
        :param mean: normalization mean where the default is ImageNet mean
        :param std: standard deviation where the defaulr is ImageNet
        :param train: bool indicating where this is a training dataset because validation
            would not augment
        """
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=resize, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "valid": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
            "test": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=resize, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "visualize_augmentations": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=resize, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ]
            ),
        }
        if uniform_augment:
            self.data_transform["train"].transforms.insert(0, UniformAugment())
            self.data_transform["test"].transforms.insert(0, UniformAugment())
            self.data_transform["visualize_augmentations"].transforms.insert(
                0, UniformAugment()
            )

    def __call__(self, img: PIL.Image, phase: bool):
        # Performs the transformations of the image
        return self.data_transform[phase](img=img)


class MelanomaDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        info_dataframe: pd.DataFrame,
        transform: UniformAugment = None,
        phase: str = "train",
        external_base_dir: str = None,
    ):
        """
        :param base_dir: this is the directory that the images 
            contained in the dataframe will be found in
        :param info_dataframe: the dataframe containing patient data and
            files names of the images
        :param transform: UniformAugment object to perform augmentations
        :param phase: which phase of training this is
        "param external_base_dir: if external data is being used then this
            is where the files will be found
        """
        assert phase in [
            "train",
            "valid",
            "test",
            "visualize_augmentations",
        ], "Phase must be one of 'train', 'valid', 'test', 'visualize_augmentations'"
        self.base_dir = base_dir
        self.info = info_dataframe
        self.transform = transform
        self.phase = phase
        self.external_base_dir = external_base_dir

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index: int):
        if self.external_base_dir != None:
            if self.info.loc[index, "tfrecord"] > 15:
                p = Path(
                    self.external_base_dir, self.info.loc[index, "image_name"] + ".jpg"
                )
            else:
                p = Path(self.base_dir, self.info.loc[index, "image_name"] + ".jpg")
        else:
            p = Path(self.base_dir, self.info.loc[index, "image_name"] + ".jpg")
        img = Image.open(p)
        img_transformed = self.transform(img, self.phase)
        if self.phase in ["train", "valid"]:
            return {
                "inputs": img_transformed,
                "labels": torch.tensor(
                    self.info.loc[index, "target"], dtype=torch.int64
                ),
            }
        else:
            return {
                "inputs": img_transformed,
            }


def format_tabular(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame = None, count: bool = False,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Takes in the dataframes containing all metadata and dummifies the categories and removes the
    filenames and patientnames before returning X and y to be fit
    :param train_raw: dataframe of tabular training data
    :param test_raw: dataframe same as above but missing target column, for testing
    :param count: whether to include a column containing the number of times
        each patient id occurs in the dataframe
    """
    # Deal with NAs first
    train_raw.loc[:, "age_approx"].fillna(0, inplace=True)
    train_raw.loc[:, "anatom_site_general_challenge"].fillna("NA", inplace=True)

    # Get just the columns wanted
    train = train_raw[["sex", "age_approx", "width", "height"]].copy()

    # Convert the sex from categorical to numerical
    train["sex"] = train["sex"].apply(lambda x: 1.0 if x == "male" else 0.0)
    train.loc[:, "sex"].fillna(-1, inplace=True)

    # Create dummy variables of other categorical columns
    train_dummies = pd.get_dummies(data=train_raw["anatom_site_general_challenge"])
    train = pd.concat([train, train_dummies], axis=1)

    # Create the count column if asked for
    if count:
        train["patient_count"] = train_raw["patient_id"].map(
            train_raw["patient_id"].value_counts()
        )

    # We do all the same for test set if it was given
    if type(test_raw) == pd.DataFrame:
        test_raw.loc[:, "age_approx"].fillna(0, inplace=True)
        test_raw.loc[:, "anatom_site_general_challenge"].fillna("NA", inplace=True)
        test = test_raw[["sex", "age_approx", "width", "height"]].copy()
        test["sex"] = test["sex"].apply(lambda x: 1.0 if x == "male" else 0.0)
        test.loc[:, "sex"].fillna(-1, inplace=True)
        test_dummies = pd.get_dummies(data=test_raw["anatom_site_general_challenge"])
        test = pd.concat([test, test_dummies], axis=1)
        if count:
            test["patient_count"] = test_raw["patient_id"].map(
                test_raw["patient_id"].value_counts()
            )

        return train, test, train_raw["target"]

    return train, train_raw["target"]

