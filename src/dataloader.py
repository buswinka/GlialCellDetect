import torch
import matplotlib.pyplot as plt
import os.path
import PIL
import numpy as np
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree
import glob
import pandas
import skimage.io
import skimage.draw
import json
from typing import Optional, Union, Dict, List, Callable
import torchvision.transforms.functional as TF
import skimage.io as io

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


class FasterRCNNData(Dataset):
    def __init__(self, basedir: str, transforms: Callable) -> None:

        self.files = glob.glob(os.path.join(basedir, '*.xml'))
        print(self.files)

        self.transforms = transforms
        self.masks = []
        self.images = []
        self.labels = []
        self.boxes = []

        for f in self.files:
            image_path = os.path.splitext(f)[0] + '.png'
            image = TF.to_tensor(PIL.Image.open(image_path)).pin_memory()

            tree = xml.etree.ElementTree.parse(f)
            root = tree.getroot()

            box_from_text = lambda a: [int(a[0].text), int(a[1].text), int(a[2].text),
                                       int(a[3].text)]

            im_shape = [image.shape[1], image.shape[2]]

            # Just because you CAN do a list comprehension, doesnt mean you SHOULD
            class_labels = torch.tensor([self._get_class_label(cls.text) for c in root.iter('object') for cls in c.iter('name')])
            bbox_loc = torch.tensor([box_from_text(a) for c in root.iter('object') for a in c.iter('bndbox')])
            mask = torch.cat([self._infer_mask_from_box(b, im_shape).unsqueeze(0) for b in bbox_loc], dim=0)

            ind = torch.logical_not(torch.isnan(class_labels))

            if ind.sum() == 0:
                continue

            class_labels = class_labels[ind].type(torch.int64)
            bbox_loc = bbox_loc[ind, :]
            mask = mask[ind, :, :]

            self.images.append(image.cuda())
            self.boxes.append(bbox_loc.pin_memory().cuda())
            self.labels.append(class_labels.pin_memory().cuda())
            self.masks.append(mask.cuda())

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        data_dict = {'boxes': self.boxes[item].clone(),
                     'labels': self.labels[item].clone(),
                     'masks': self.masks[item].clone(),
                     'image': self.images[item].clone()}

        data_dict = self.transforms(data_dict)

        return data_dict['image'], data_dict

    @staticmethod
    def _get_class_label(label_text: str) -> int:
        label = 0
        if label_text == 'active':
            label = 1  # 1
        elif label_text == 'resting':
            label = 2
        elif label_text == 'Junk' or label_text == 'junk':
            label = float('nan')  # 7
        else:
            raise ValueError(f'Unidentified Label in XML file {label_text}')

        return label

    @staticmethod
    def _infer_mask_from_box(box: torch.Tensor, shape: Union[list, tuple, torch.Tensor]) -> torch.Tensor:
        mask = torch.zeros(shape)
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        mask[y1:y2, x1:x2] = 1
        return mask

    def __len__(self):
        return len(self.images)


def _calculate_mask(d: Dict[str, list], mask: torch.Tensor) -> torch.Tensor:
    x = d['all_points_x']
    y = d['all_points_y']
    xx, yy = skimage.draw.polygon(x, y)
    mask[yy, xx] = 1
    return mask


def _calculate_box(d: Dict[str, list]) -> torch.Tensor:
    x = d['all_points_x']
    y = d['all_points_y']
    return torch.tensor([np.min(x), np.min(y), np.max(x), np.max(y)])


def _get_label(d: Dict[str, list]) -> torch.Tensor:
    if 'stereocillia' in d:
        label = int(d['stereocillia'])
    elif 'stereocilia' in d:
        label = int(d['stereocilia'])
    else:
        label = None
    return label


def _get_masks_from_csv(image_names, data_frame):
    for im_name in image_names:
        df = data_frame[data_frame['save_name'] == im_name]
        im_path = os.path.join(basedir, im_name)

        if len(df) <= 1 or not os.path.exists(im_path):  # some dataframes will contain no data... skip
            continue

        image = TF.to_tensor(PIL.Image.open(im_path)).pin_memory()

        mask = torch.zeros((len(df), image.shape[1], image.shape[2]), dtype=torch.uint8)

        region_shape_attributes = df['region_shape_attributes'].to_list()
        region_attributes = df['region_attributes'].to_list()

        # List comprehensions are... beautiful? ;D
        # Future chris says this is the worst code he's ever written
        mask = torch.cat(
            [_calculate_mask(json.loads(d), m).unsqueeze(0) for d, m in zip(region_shape_attributes, mask)],
            dim=0).int()
        boxes = torch.cat([_calculate_box(json.loads(d)).unsqueeze(0) for d in region_shape_attributes], dim=0)
        labels = torch.tensor([_get_label(json.loads(d)) for d in region_attributes])

        for l in range(mask.shape[0]):
            if mask[l, :, :].max() == 0:
                raise ValueError('Mask is jank')
