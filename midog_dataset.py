import openslide
from pathlib import Path
import numpy as np
from random import randint
from torch.utils.data import Dataset
import torch

from typing import Union
import viz_utils
from sklearn.feature_extraction import image
from nms import nms, calculate_slice_bboxes

should_normalise = False


class SlideContainer:

    def __init__(self, file: Union[Path, str], image_id: int, y, level: int = 0, width: int = 256, height: int = 256):
        self.file = file # Name of file
        self.image_id = image_id
        self.slide = openslide.open_slide(str(file)) # File
        self.width = width
        self.height = height
        self.down_factor = self.slide.level_downsamples[level]
        self.targets = y
        self.classes = list(set(self.targets[1]))

        if level is None:
            level = self.slide.level_count - 1
        self.level = level

    def get_patch(self,  x: int = 0, y: int = 0):
        try:
            arr = np.copy(np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self.level, size=(self.width, self.height)))[:, :, :3])
            return arr

        except OSError as error:
            return np.zeros((self.width, self.height, 3), dtype=np.uint8)

    @property
    def shape(self):
        return self.width, self.height

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self.level]

    def __str__(self):
        return str(self.file)

    def get_new_train_coordinates(self, sample_func=None):
        # use passed sampling method
        if callable(sample_func):
            return sample_func(self.targets, **{"classes": self.classes, "shape": self.shape,
                                                     "level_dimensions": self.slide.level_dimensions,
                                                     "level": self.level})

        # use default sampling method
        width, height = self.slide.level_dimensions[self.level]
        return np.random.randint(0, width - self.shape[0]), np.random.randint(0, height - self.shape[1])


class MIDOGTrainDataset(Dataset):

    def __init__(self, list_containers: list[SlideContainer], patches_per_slide: int = 10, transform=None, sample_func=None) -> None:
        super().__init__()
        self.list_containers = list_containers
        self.patches_per_slide = patches_per_slide
        self.transform = transform
        self.sample_func = sample_func
        self.patches = [None] * len(self.list_containers) * self.patches_per_slide

    def __len__(self):
        return len(self.list_containers)*self.patches_per_slide

    def __getitem__(self, idx):
        idx_slide = idx % len(self.list_containers)
        cur_image_container = self.list_containers[idx_slide]
        train_coordinates = cur_image_container.get_new_train_coordinates(self.sample_func)
        self.patches[idx] = train_coordinates
        return get_patch_w_labels(cur_image_container, self.transform, *train_coordinates)
    

class MIDOGTestDataset(Dataset):

    def __init__(self, container: SlideContainer, nms_threshold=0.4, transform=None, ):
        self.nms_threshold = nms_threshold
        self.container = container
        self.patches = get_patches(container, self.nms_threshold)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)



    def __getitem__(self, idx):
        patch = self.patches[idx]
        x = patch[0]
        y = patch[1]
        return get_patch_w_labels(self.container, self.transform,  x, y)

    def get_slide_labels_as_dict(self) -> dict:
        bboxes, labels = self.container.targets
        bboxes = bboxes.reshape((-1, 4))
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([self.container.image_id]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
            'area': torch.as_tensor(area, dtype=torch.float32)
        }

        return targets

    def local_to_global(self, idx, bboxes:torch.Tensor) -> torch.Tensor:
        patch = self.patches[idx]
        x0 = patch[0]
        y0 = patch[1]

        bboxes_global = []

        for i, box in enumerate(bboxes):
            x_min = int(box[0] + x0)
            x_max = int(box[2] + x0)
            y_min = int(box[1] + y0)
            y_max = int(box[3] + y0)
            global_box = torch.tensor([x_min, y_min, x_max, y_max])
            bboxes_global.append(global_box)
        if len(bboxes_global) >0:
            return torch.stack(bboxes_global)
        else:
            return torch.tensor(bboxes_global)


def get_patch_w_labels(cur_container: SlideContainer, transform, x: int = 0, y: int = 0):
    patch = cur_container.get_patch(x, y)

    bboxes, labels = cur_container.targets
    h, w = cur_container.shape

    bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
    labels = np.array(labels)

    area = np.empty([0])

    if len(labels) > 0:
        bboxes, labels = viz_utils.filter_bboxes(bboxes, labels, x, y, w, h)
        if transform:
            transformed = transform(image=patch, bboxes=bboxes, class_labels=labels)
            patch = transformed['image']
            bboxes = np.array(transformed['bboxes'])
            labels = transformed['class_labels']

        if len(bboxes) > 0:
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    else:
        if transform:
            patch = transform(image=patch, bboxes=bboxes, class_labels=labels)['image']

    bboxes = bboxes.reshape((-1, 4))

    # following the label definition described here:
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#torchvision-object-detection-finetuning-tutorial
    targets = {
        'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([cur_container.image_id]),
        'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        'area': torch.as_tensor(area, dtype=torch.float32)
    }
    # print(targets)
    patch_as_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float()
    return min_max_normalization(patch_as_tensor), targets


def min_max_normalization(x):
    if should_normalise:
        z = (x - x.mean(axis=(1,2), keepdims=True)) / x.std(axis=(1,2), keepdims=True)
    else:
        z = x/255.0
    return z

def get_patches(container, nms_threshold):
    slide = container.slide
    level = container.level
    size = slide.level_dimensions[level]

    patch_width =container.width
    patch_height = container.height
    patches = calculate_slice_bboxes(size[0], size[1], patch_width, patch_height, nms_threshold)
    return patches
