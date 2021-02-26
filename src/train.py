import src.model
from src.dataloader import MaskData, KeypointData
import src.transforms as t
import src.utils
import torch.optim

import numpy as np
import time
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

def train_mask_rcnn(data=None, epochs: int = None, lr: float = None,
                    pretrained: str = None):

    mask_rcnn = src.model.mask_rcnn

    if not isinstance(pretrained, str) and pretrained is not None:
        raise ValueError(f'Argument "pretrained" must be a path to a valid mask file, '
                         f'not {pretrained} with type {type(pretrained)}')
    if epochs is None:
        epochs = 500

    if lr is None:
        lr = 1e-5

    if pretrained is not None:
        print('Loading...')
        mask_rcnn.load_state_dict(torch.load(pretrained))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mask_rcnn.train().to(device)
    optimizer = torch.optim.Adam(mask_rcnn.parameters(), lr=lr)

    epoch_range = trange(epochs, desc='Loss: {1.00000}', leave=True)

    out_str = ''
    if len(data) < 1:
        raise ValueError('No available train data')

    for e in epoch_range:
        epoch_loss = []
        for image, data_dict in data:
            optimizer.zero_grad()
            loss = mask_rcnn(image.unsqueeze(0), [data_dict])
            losses = 0
            for key in loss:
                losses += loss[key]
            losses.backward()
            epoch_loss.append(losses.item())
            optimizer.step()

        epoch_range.desc = 'Loss: ' + '{:.5f}'.format(torch.tensor(epoch_loss).mean().item())

    torch.save(mask_rcnn.state_dict(), 'models/mask_rcnn.mdl')

    return None


def train_keypoint_rcnn(data=None, epochs: int = None, lr: float = 1e-5, pretrained: str = None):

    model = src.model.keypoint_rcnn

    if not isinstance(pretrained, str) and pretrained is not None:
        raise ValueError(f'Argument "pretrained" must be a path to a valid mask file, '
                         f'not {pretrained} with type {type(pretrained)}')
    if epochs is None:
        epochs = 500

    if pretrained is not None:
        print('Loading...')
        model.load_state_dict(torch.load(pretrained))

    if torch.cuda.is_available(): device = 'cuda:0'
    else: device = 'cpu'

    # tests = KeypointData('/media/DataStorage/Dropbox (Partners HealthCare)/DetectStereocillia/data/keypoint_train_data')
    model = model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for e in range(epochs):
        epoch_loss = []
        time_1 = time.clock_gettime_ns(1)
        for image, data_dict in data:
            for key in data_dict:
                data_dict[key] = data_dict[key].to(device)
            assert image.shape[1] == 3

            optimizer.zero_grad()
            loss = model(image.to(device), [data_dict])
            losses = 0
            for key in loss:
                losses += loss[key]
            losses.backward()
            epoch_loss.append(losses.item())
            optimizer.step()
        time_2 = time.clock_gettime_ns(1)

        delta_time = np.round((np.abs(time_2-time_1) / 1e9)/60, decimals=2)

            #  --------- This is purely to output a nice bar for training --------- #
        if e % 5 == 0:
            if e > 0:
                print('\b \b' * len(out_str), end='')
            progress_bar = '[' + '█' * +int(np.round(e / epochs, decimals=1) * 10) + \
                           ' ' * int(
                (10 - np.round(e / epochs, decimals=1) * 10)) + f'] {np.round(e / epochs, decimals=3)}%'

            out_str = f'epoch: {e} ' + progress_bar + f'| time remaining: {delta_time * (epochs-e)} min | epoch loss: {torch.tensor(epoch_loss).mean().item()}'
            print(out_str, end='')

        # If its the final epoch print out final string
        elif e == epochs - 1:
            print('\b \b' * len(out_str), end='')
            progress_bar = '[' + '█' * 10 + f'] {1.0}'
            out_str = f'epoch: {epochs} ' + progress_bar + f'| time remaining: {0} min | epoch loss: {torch.tensor(epoch_loss).mean().item()}'
            print(out_str)

        torch.save(model.state_dict(), 'models/keypoint_rcnn.mdl')

    model.eval()
    out = model(image.unsqueeze(0).cuda())