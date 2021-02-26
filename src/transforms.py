import torch
import torchvision.transforms.functional
from PIL.Image import Image
import numpy as np
from typing import Dict, Tuple, Union, List


class random_v_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.vflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = self.fun(data_dict['image'])
            data_dict['masks'] = self.fun(data_dict['masks'])

        return data_dict


class random_h_flip:
    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.hflip)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly flips the mask vertically.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        if torch.rand(1) < self.rate:
            data_dict['image'] = self.fun(data_dict['image'])
            data_dict['masks'] = self.fun(data_dict['masks'])

        return data_dict


class normalize:
    def __init__(self, mean: List[float] = [0.5], std: List[float] = [0.5]) -> None:
        self.mean = mean
        self.std = std
        self.fun = torch.jit.script(torchvision.transforms.functional.normalize)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        data_dict['image'] = self.fun(data_dict['image'], self.mean, self.std)
        return data_dict


class gaussian_blur:
    def __init__(self, kernel_targets: torch.Tensor = torch.tensor([3, 5, 7]), rate: float = 0.5) -> None:
        self.kernel_targets = kernel_targets
        self.rate = rate
        self.fun = torch.jit.script(torchvision.transforms.functional.gaussian_blur)

    def __call__(self, data_dict):
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.rand(1) < self.rate:
            kern = self.kernel_targets[int(torch.randint(0, len(self.kernel_targets), (1, 1)).item())].item()
            data_dict['image'] = self.fun(data_dict['image'], [kern, kern])
        return data_dict


class random_resize:
    def __init__(self, rate: float = 0.5, scale: tuple = (300, 1440)) -> None:
        self.rate = rate
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Randomly resizes an mask

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'mask' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """
        if torch.rand(1) < self.rate:
            size = torch.randint(self.scale[0], self.scale[1], (1, 1)).item()
            data_dict['image'] = torchvision.transforms.functional.resize(data_dict['image'], size)
            data_dict['masks'] = torchvision.transforms.functional.resize(data_dict['masks'], size)

        return data_dict


class adjust_brightness:
    def __init__(self, rate: float = 0.5, range_brightness: Tuple[float, float] = (0.3, 1.7)) -> None:
        self.rate = rate
        self.range = range_brightness
        self.fun = torch.jit.script(torchvision.transforms.functional.adjust_brightness)

    def __call__(self, data_dict):

        val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1])
        if torch.rand(1) < self.rate:
            data_dict['image'] = self.fun(data_dict['image'], val.item())

        return data_dict


# needs docstring
class adjust_contrast:
    def __init__(self, rate: float = 0.5, range_contrast: Tuple[float, float] = (.3, 1.7)) -> None:
        self.rate = rate
        self.range = range_contrast
        self.fun = torch.jit.script(torchvision.transforms.functional.adjust_brightness)

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if torch.rand(1) < self.rate:
            val = torch.FloatTensor(1).uniform_(self.range[0], self.range[1])  # .to(image.device)
            data_dict['image'] = torchvision.transforms.functional.adjust_contrast(data_dict['image'], val.to(data_dict['image'].device))

        return data_dict


# needs docstring
class random_affine:
    def __init__(self, rate: float = 0.5, angle: Tuple[int, int] = (-180, 180),
                 shear: Tuple[int, int] = (-25, 25), scale: Tuple[float, float] = (0.9, 1.5)) -> None:
        self.rate = rate
        self.angle = angle
        self.shear = shear
        self.scale = scale

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if torch.rand(1) < self.rate:
            angle = torch.FloatTensor(1).uniform_(self.angle[0], self.angle[1])
            shear = torch.FloatTensor(1).uniform_(self.shear[0], self.shear[1])
            scale = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1])
            translate = torch.tensor([0, 0])

            data_dict['image'] = _affine(data_dict['image'], angle, translate, scale, shear)

            # mask = data_dict['masks']  # [C, X, Y ,Z]
            # num_stereocilia = mask.shape[0]
            #
            # for i in range(num_stereocilia):
            #     mask[i, ...] *= i + 1
            # mask = mask.argmax(dim=0).unsqueeze(0)
            # print(mask.shape)

            data_dict['masks'] = _affine(data_dict['masks'], angle, translate, scale, shear)

        return data_dict


class to_cuda:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move every element in a dict containing torch tensor to cuda.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader
        :return: Dict[str, torch.Tensor]
        """
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        return data_dict


class to_tensor:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, Union[torch.Tensor, Image, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Convert a PIL image or numpy.ndarray to a torch.Tensor

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader
        :return: Dict[str, torch.Tensor]
        """
        data_dict['image'] = torchvision.transforms.functional.to_tensor(data_dict['image'])
        return data_dict


class correct_boxes:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Other geometric transforms may have removed some of the visible stereocillia. We use this transform to infer new
        bounding boxes from the old masks and remove instances (I) where  there was no segmentation mask.

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y] where C is the number of colors, X,Y are the mask height and width
            'masks' : torch.Tensor of size [I, X, Y] where I is the number of identifiable objects in the mask
            'boxes' : torch.Tensor of size [I, 4] where each box is [x1, y1, x2, y2]
            'labels' : torch.Tensor of size [I] class label for each instance

        :return: Dict[str, torch.Tensor]
        """

        return _correct_box(image=data_dict['image'], masks=data_dict['masks'], labels=data_dict['labels'])


class stack_image:
    def __init__(self):
        pass

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data_dict['image'] = torch.cat((data_dict['image'], data_dict['image'], data_dict['image']), dim=0)
        return data_dict


@torch.jit.script
def get_box_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Returns the bounding box for a particular segmentation mask
    :param: mask torch.Tensor[X,Y] some mask where 0 is background and !0 is a segmentation mask
    :return: torch.Tensor[4] coordinates of the box surrounding the segmentation mask [x1, y1, x2, y2]
    """
    ind = torch.nonzero(mask)

    if ind.shape[0] == 0:
        box = torch.tensor([0, 0, 0, 0]).to(mask.device)

    else:
        box = torch.empty(4).to(mask.device)
        x = ind[:, 1]
        y = ind[:, 0]
        torch.stack((torch.min(x), torch.min(y), torch.max(x), torch.max(y)), out=box)

    return box


@torch.jit.script
def _correct_box(image: torch.Tensor,  masks: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Infers new boxes from a transformed ground truth mask

    :param image: torch.Tensor[float] of shape [C, X, Y] -> C: number of color channels
    :param masks: torch.Tensor[float] of shape [I, X, Y] -> I: number of instances
    :param labels: torch.Tensor[float] of shape [I]
    :return: Dict[str, torch.Tensor] with keys 'image', 'masks', 'boxes' 'labels'
    """
    # boxes = torch.zeros((4, masks.shape[0]))
    boxes = torch.cat([get_box_from_mask(m).unsqueeze(0) for m in masks], dim=0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ind = torch.tensor([a.item() > 0 for a in area], dtype=torch.bool)

    return {'image': image, 'masks': masks[ind, :, :], 'boxes': boxes[ind, :], 'labels': labels[ind]}


@torch.jit.script
def _affine(img: torch.Tensor, angle: torch.Tensor, translate: torch.Tensor, scale: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    """

    :param img:
    :param angle:
    :param translate:
    :param scale:
    :param shear:
    :return:
    """
    # raise NotImplementedError('Current Implementation is not memory Efficient  - todo: perform affine on argmax of masks'+
    #                           'Then re-infer masks from the thingy')
    angle = float(angle.item())
    scale = float(scale.item())
    shear = [float(shear.item())]
    translate_list = [int(translate[0].item()), int(translate[1].item())]
    return torchvision.transforms.functional.affine(img, angle, translate_list, scale, shear)
