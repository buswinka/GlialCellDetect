import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import PIL.Image
import skimage.io
import skimage.color
from typing import Union, Dict, List


def show_box_pred_simple(image, boxes):

    # x1, y1, x2, y2

    plt.imshow(image)
    # plt.tight_layout()

    for i, box in enumerate(boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1, x2], [y2, y2], 'r', lw=0.5)
        plt.plot([x1, x2], [y1, y1], 'r', lw=0.5)
        plt.plot([x1, x1], [y1, y2], 'r', lw=0.5)
        plt.plot([x2, x2], [y1, y2], 'r', lw=0.5)
    plt.show()


def render_mask(image: torch.Tensor, model_output: dict, threshold: float) -> None:
    image = image[0, 0, ...].detach().cpu().numpy() if image.ndim == 4 else image[0, ...].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()
    masks = model_output['masks'].detach().cpu().numpy() > 0.5

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    colormask = np.zeros((masks.shape[-2], masks.shape[-1], 3))

    simple_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i, box in enumerate(boxes):
        try:
            if model_output['scores'][i] < threshold:
                continue
        except KeyError:
            pass

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1, x2], [y2, y2], 'r', lw=0.5)
        plt.plot([x1, x2], [y1, y1], 'r', lw=0.5)
        plt.plot([x1, x1], [y1, y2], 'r', lw=0.5)
        plt.plot([x2, x2], [y1, y2], 'r', lw=0.5)

        if masks.ndim == 4:
            colormask[..., 0][masks[i, 0, ...] > 0] = simple_colors[model_output['labels'][i] - 1][0]
            colormask[..., 1][masks[i, 0, ...] > 0] = simple_colors[model_output['labels'][i] - 1][1]
            colormask[..., 2][masks[i, 0, ...] > 0] = simple_colors[model_output['labels'][i] - 1][2]
        else:
            colormask[..., 0][masks[i, ...] > 0] = simple_colors[model_output['labels'][i] - 1][0]
            colormask[..., 1][masks[i, ...] > 0] = simple_colors[model_output['labels'][i] - 1][1]
            colormask[..., 2][masks[i, ...] > 0] = simple_colors[model_output['labels'][i] - 1][2]
    plt.imshow(colormask, alpha=0.35)
    plt.show()


def render_boxes(image: torch.Tensor, model_output: dict, threshold: float) -> None:
    image = image.transpose(0, -1).transpose(0,1).detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    for i, box in enumerate(boxes):
        if model_output['scores'][i] < threshold:
            continue
        label = model_output['labels'][i]

        c = ['nul', 'r', 'y', 'y', 'w']

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1, x2], [y2, y2], c[label], lw=1.5)
        plt.plot([x1, x2], [y1, y1], c[label], lw=1.5)
        plt.plot([x1, x1], [y1, y2], c[label], lw=1.5)
        plt.plot([x2, x2], [y1, y2], c[label], lw=1.5)

    plt.show()


def render_keypoints(image: torch.Tensor, model_output: dict, threshold: float) -> None:
    image = image[0, 0, :, :].detach().cpu().numpy()
    boxes = model_output['boxes'].detach().cpu().numpy()
    keypoints = model_output['keypoints'].detach().cpu().numpy()

    plt.imshow(image, cmap='Greys_r')
    plt.tight_layout()

    for i, box in enumerate(boxes):
        if model_output['scores'][i] < threshold:
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        # plt.plot([x1,x2],[y2,y2],'r', lw=0.5)
        # plt.plot([x1,x2],[y1,y1],'r', lw=0.5)
        # plt.plot([x1,x1],[y1,y2],'r', lw=0.5)
        # plt.plot([x2,x2],[y1,y2],'r', lw=0.5)

        x = keypoints[i, :, 0]
        y = keypoints[i, :, 1]
        plt.plot(x, y, 'b-', alpha=0.5)
        plt.plot(x, y, 'b.', alpha=0.5)

        plt.plot()

    plt.show()


class image:
    def __init__(self, base_im: Union[torch.Tensor, np.ndarray]) -> None:

        if isinstance(base_im, np.ndarray):
            self.image = torch.from_numpy(base_im)
        else:
            self.image = base_im

        self.colormask = np.zeros((3, self.image.shape[-2], self.image.shape[-1]))
        self.simple_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    def add_partial_mask(self, x: int, y: int, model_output: Dict[str, torch.Tensor], threshold: float) -> None:
        """
        Add a eval chunk to the image struct. Allows you to crop and separately analyze large images then merge them
        back into one image.

        :param x: x coord of a crop of the base image
        :param y: y coord of a crop of the base image
        :param model_output: output dict of a pytorch model
            has keys: 'masks', 'scores', 'labels'
        :param threshold: lower bound score threshold where we dont plot model output

        :return: None
        """

        masks = model_output['masks'].detach().cpu().numpy()

        for i in range(masks.shape[0]):
            try:
                if model_output['scores'][i] < threshold:
                    continue
            except KeyError:
                pass

            if masks.ndim == 4:
                self.colormask[0, x:x + masks.shape[-2], y:y + masks.shape[-1]][masks[i, 0, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][0]
                self.colormask[1, x:x + masks.shape[-2], y:y + masks.shape[-1]][masks[i, 0, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][1]
                self.colormask[2, x:x + masks.shape[-2], y:y + masks.shape[-1]][masks[i, 0, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][2]
            else:
                self.colormask[x:x + masks.shape[1] - 1, y:y + masks.shape[2] - 1, 0][masks[i, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][0]
                self.colormask[x:x + masks.shape[1] - 1, y:y + masks.shape[2] - 1, 1][masks[i, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][1]
                self.colormask[x:x + masks.shape[1] - 1, y:y + masks.shape[2] - 1, 2][masks[i, :, :] > 0.5] = \
                    self.simple_colors[model_output['labels'][i] - 1][2]

    def render(self, save_name: str = 'my_image.png') -> None:
        """
        Saves output from render_mat to a file

        :param save_name: save file path
        :return:
        """
        out = self.render_mat()
        skimage.io.imsave(save_name, out.transpose((1, 2, 0)))
        return None

    def render_mat(self) -> np.ndarray:
        """
        Return rendered image as a numpy matrix
        :return:
        """
        im = self.image.numpy()[0, :, :, :].astype(np.float)
        cm = self.colormask.astype(np.float)
        out = cv2.addWeighted(cm, 0.35, im, 1 - 0.35, 0)
        out = skimage.img_as_ubyte(out)
        return out


def color_from_ind(i: int) -> np.ndarray:
    """
    Take in some number and always generate a unique color from that number.
    Quick AF
    :param i:
    :return:
    """
    np.random.seed(i)
    return np.random.random(3)


def calculate_indexes(pad_size: int, eval_image_size: int, image_shape: int, padded_image_shape: int) -> List[List[int]]:
    """
    This calculates indexes for the complete evaluation of an arbitrarily large image by unet.
    each index is offset by eval_image_size, but has a width of eval_image_size + pad_size * 2.
    Unet needs padding on each side of the evaluation to ensure only full convolutions are used
    in generation of the final mask. If the algorithm cannot evenly create indexes for
    padded_image_shape, an additional index is added at the end of equal size.

    :param pad_size: int corresponding to the amount of padding on each side of the
                     padded image
    :param eval_image_size: int corresponding to the shape of the image to be used for
                            the final mask
    :param image_shape: int Shape of image before padding is applied

    :param padded_image_shape: int Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size > image_shape:
        return [[0, image_shape]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(f'Calculate_indexes has incorrect values {pad_size} | {image_shape} | {eval_image_size}:\n'
                           f'You are likely trying to have a chunk smaller than the set evaluation image size. '
                           'Please decrease number of chunks.')
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1])
        z2 = int(z-1) + (2 * pad_size)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size+pad_size * 2), padded_image_shape])
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + pad_size * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind


def merge_results_dict(results: Dict[str, List[Union[torch.Tensor, List[int]]]]) -> Dict[str, torch.Tensor]:
    """
    Merge results of multiple evaluations of faster_rcnn

    :param results:
    :return:
    """
    first = True
    for i, data in enumerate(results['out_dict']):  # should be a list here
        if first and data['boxes'].shape[0] != 0:
            labels = data['labels']
            scores = data['scores']
            boxes = data['boxes']
            boxes[:, [0, 2]] += results['x'][i][0]
            boxes[:, [1, 3]] += results['y'][i][0]
            first = False

        elif not first and data['boxes'].shape != 0:
            labels = torch.cat((labels, data['labels']), dim=0)
            scores = torch.cat((scores, data['scores']), dim=0)
            b = data['boxes']
            b[:, [0, 2]] += results['x'][i][0]
            b[:, [1, 3]] += results['y'][i][0]
            boxes = torch.cat((boxes, b), dim=0)

    return {'scores': scores, 'boxes': boxes, 'labels': labels}