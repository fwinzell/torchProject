import os
import cv2
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

from syngan.models import GenUnet
from syngan.pretrain import parse_config

from prostate_clr.evaluation import image_to_patches, preprocess, get_tiles


def tiling(tiles, image_shape, tile_sz=224, patch_sz=256):
    image = np.zeros(shape=image_shape)
    pad = int((patch_sz - tile_sz) / 2)
    nbr_xpatches = math.ceil(image_shape[1] / tile_sz)
    nbr_ypatches = math.ceil(image_shape[0] / tile_sz)
    ycor = 0
    count = 0
    for i in range(nbr_ypatches):
        xcor = 0
        for j in range(nbr_xpatches):
            image[ycor:ycor + tile_sz, xcor:xcor + tile_sz, :] = tiles[count, pad:pad + tile_sz, pad:pad + tile_sz, :]
            count += 1
            xcor = xcor + tile_sz
            if xcor > image_shape[1] - tile_sz:
                xcor = image_shape[1] - tile_sz
        ycor = ycor + tile_sz
        if ycor > image_shape[0] - tile_sz:
            ycor = image_shape[0] - tile_sz

    return image


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = GenUnet(parse_config()).to(device)

    model_weights = torch.load("/home/fi5666wi/Documents/Python/saved_models/syngan/pt_unet/pt_49_model.pth")
    model.load_state_dict(model_weights, strict=True)

    eval_path = '/home/fi5666wi/Documents/Prostate images/Eval'
    gt_image = cv2.imread(os.path.join(eval_path, 'image_slic/cropped_image_slic-annotated.png'))
    ds = get_tiles(gt_image)  # image_to_patches(gt_image, overlap=0)

    output = []
    for idx in range(len(ds)):
        mask = ds[idx]
        inp = preprocess(mask)
        x = torch.from_numpy(inp).to(device)  # to torch, send to device
        with torch.no_grad():
            y = model(x)

        res = np.squeeze(y.cpu().numpy())
        res = np.uint8(res*255)
        res = np.moveaxis(res, source=0, destination=-1)
        output.append(res)

        #cv2.imshow('Result', res)
        #cv2.imshow('Mask', mask)
        #cv2.waitKey(1000)
    #cv2.destroyAllWindows()

    synthetic = tiling(np.array(output), gt_image.shape)
    synthetic = np.uint8(synthetic)
    cv2.imshow('Synthetic', synthetic)
    cv2.waitKey(0)






