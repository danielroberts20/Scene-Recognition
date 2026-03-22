import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from scenes import Scene
from util import get_testing_images


class Run3TestDataset(Dataset):
    def __init__(self, image_dir="testing", transform=None):

        self.image_dir = image_dir
        self.transform = transform
        self.image_names = get_testing_images(image_dir)

    def __len__(self):
        # Return the total number of samples
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load the image and its corresponding label
        img_name = self.image_names[idx]
        image = Image.open(img_name).convert("L")
        image = Image.merge("RGB", (image, image, image))

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, img_name

class Run3DatasetGenerator:

    train_data = []
    validate_data = []

    def __init__(self,
                 directory: str,
                 validation_size: float = None):

        if validation_size is None:
            validation_size = 0.
        elif not (0 <= validation_size <= 1):
            raise ValueError("Validation size must be between 0 and 1.")

        # Iterate through each subfolder in the base folder
        for scene in os.listdir(directory):
            subfolder_path = os.path.join(directory, scene)

            if os.path.isdir(subfolder_path):  # Ensure it's a directory
                # Get all image files (you can add filtering for specific file extensions if needed)
                image_paths = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

                # Sort paths to ensure a consistent order (optional, but recommended)
                image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

                # Number of images in the subfolder
                total_images = len(image_paths)

                # Calculate how many images should be in the first and second list
                num_second_list = int(total_images * validation_size)
                num_first_list = total_images - num_second_list

                # Add the first (1 - percentage)% of images to the first list
                self.train_data.extend(image_paths[:num_first_list])

                # Add the remaining percentage% of images to the second list
                self.validate_data.extend(image_paths[num_first_list:])

    def train(self, transform = None):
        return Run3Dataset(self.train_data, transform)

    def validate(self, transform = None):
        return Run3Dataset(self.validate_data, transform)

class Run3Dataset(Dataset):

    def __init__(self,
                 data,
                 transform=None):

        self.data = data
        self.transform = transform
        self.labels = [s.out for s in Scene]

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image and its corresponding label
        img_path = self.data[idx]
        image = Image.open(img_path).convert("L")
        image = Image.merge("RGB", (image, image, image))

        label = Scene.from_path(img_path).out
        label_index = self.labels.index(label)

        label = np.zeros(len(self.labels))
        label[label_index] = 1

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, label
